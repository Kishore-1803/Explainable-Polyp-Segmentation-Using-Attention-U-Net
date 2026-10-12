import os
import io
import math
import base64
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, jsonify, make_response

# Custom modules
import sys
# variants.py imports the shared blocks as `models.attention_unet`, so src/
# must be importable as a top-level path as well as the `src` package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiments", "paper_replication"))
from variants import VariantAttentionUNet
from src.explainability.grad_cam import SegmentationGradCAM
from src.evaluation.confidence import ConfidenceScorer
from src.preprocessing.crop import get_bounding_box

app = Flask(__name__)

# ─────────────────────────────────────────────
# PyTorch Model & Grad-CAM Initialization
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[*] Initializing Attention U-Net on device: {DEVICE}")

# Best model from the replication study: full-width Attention U-Net with
# Squeeze-and-Excitation channel attention and deep supervision, trained on
# luminance-channel CLAHE. Kvasir-SEG 0.8531 DSC, CVC-ClinicDB 0.7015 zero-shot.
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints",
                               "paper_var_se_ds_ours", "best.pth")

print(f"[*] Loading model checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# train_variant.py saves a bare state_dict; older runs wrapped it in a dict.
state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

model = VariantAttentionUNet(in_channels=3, out_channels=1, base=64,
                             use_se=True, deep_supervision=True).to(DEVICE)
model.load_state_dict(state)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"[*] Attention U-Net (SE + deep supervision) loaded: {n_params:,} parameters.")

# Initialize Segmentation Grad-CAM targeting decoder up4
grad_cam = SegmentationGradCAM(model, model.up4)
print(f"[*] Segmentation Grad-CAM attached to decoder up4.")

# ─────────────────────────────────────────────
# Preprocessing & Visualization Utilities
# ─────────────────────────────────────────────

def compute_laplacian_variance(img_gray: np.ndarray) -> float:
    """Laplacian blur detection: variance below 80.0 indicates blur."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def apply_clahe_lab(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE on the L-channel in LAB color space.
    Clip limit 2.0, tile grid 8x8 — matches EXP-08 training.
    """
    img_lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def array_to_base64_jpg(img_rgb: np.ndarray, quality: int = 88) -> str:
    """Encodes an RGB numpy array into a base64 JPEG data URI."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def create_mask_overlay(clahe_rgb: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """
    Overlays the predicted binary segmentation mask onto the CLAHE image.
    Uses emerald green fill (alpha 0.5) with bright cyan contour boundary.
    """
    overlay = clahe_rgb.copy()
    if mask_bin.sum() == 0:
        return overlay

    green_fill = np.array([16, 185, 129], dtype=np.uint8)
    pos = mask_bin > 0.5
    overlay[pos] = cv2.addWeighted(clahe_rgb[pos], 0.45, np.tile(green_fill, (np.sum(pos), 1)), 0.55, 0)

    contours, _ = cv2.findContours((mask_bin * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (6, 214, 160), 2)
    return overlay


def create_gradcam_overlay(clahe_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Blends the normalized Grad-CAM heatmap with the CLAHE image using JET colormap.
    """
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(clahe_rgb, 0.5, heatmap_colored_rgb, 0.5, 0)


def estimate_heatmap_focus(heatmap: np.ndarray) -> str:
    """Estimates the dominant region of Grad-CAM attention."""
    h, w = heatmap.shape
    quads = {
        "Center":       heatmap[h//4:3*h//4, w//4:3*w//4],
        "Top-Left":     heatmap[:h//2, :w//2],
        "Top-Right":    heatmap[:h//2, w//2:],
        "Bottom-Left":  heatmap[h//2:, :w//2],
        "Bottom-Right": heatmap[h//2:, w//2:],
    }
    scores = {k: float(np.mean(v)) for k, v in quads.items()}
    best = max(scores, key=scores.get)
    return f"{best} Region"


# ─────────────────────────────────────────────
# Real End-to-End Inference Pipeline
# ─────────────────────────────────────────────

def analyze_image_with_model(pil_image: Image.Image) -> dict:
    """
    Runs the complete verified segmentation pipeline:
    1. Laplacian blur detection & specular exposure check
    2. Adaptive LAB CLAHE enhancement
    3. Attention U-Net forward pass
    4. Exact lesion area & bounding box calculation
    5. Statistically grounded confidence scoring (Entropy & Boundary margin)
    6. Decoder Grad-CAM heatmap generation
    7. Diagnostic overlay rendering
    """
    orig_rgb = np.array(pil_image.convert("RGB").resize((256, 256)), dtype=np.uint8)
    gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Quality analysis
    lap_var = compute_laplacian_variance(gray)
    is_blurred = lap_var < 80.0
    quality = "Blurred (Suboptimal)" if is_blurred else "Sharp (Diagnostic Quality)"

    brightness = float(np.mean(gray)) / 255.0
    is_overexposed = brightness > 0.85
    is_underexposed = brightness < 0.15

    # 2. CLAHE enhancement
    clahe_rgb = apply_clahe_lab(orig_rgb)
    clahe_pil = Image.fromarray(clahe_rgb)

    # 3. Tensor normalization
    tensor = TF.to_tensor(clahe_pil)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)

    # 4. Attention U-Net Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    # 5. Grad-CAM generation
    heatmaps, _ = grad_cam.generate_heatmap(tensor)
    heatmap = heatmaps[0]

    mask_np = mask.cpu().numpy().squeeze()
    lesion_area = int(np.sum(mask_np > 0.5))
    total_pixels = 256 * 256
    lesion_pct = round((lesion_area / total_pixels) * 100, 2)

    # Polyp detected if mask area >= 50 pixels (noise threshold)
    is_polyp = lesion_area >= 50

    # 6. Confidence and Uncertainty Scoring
    conf_map = ConfidenceScorer.compute_binary_confidence(logits)
    entropy_map = ConfidenceScorer.compute_entropy(logits)

    if is_polyp:
        conf_val = float(ConfidenceScorer.aggregate_segmentation_confidence(conf_map[0, 0], mask[0, 0]))
        entropy_val = float(entropy_map[0, 0][mask[0, 0] > 0.5].mean().item())
        status_label = "Polyp Detected"
        risk = "High Risk" if lesion_area > 3000 else "Moderate Risk"
        ymin, ymax, xmin, xmax = get_bounding_box(mask[0, 0], padding=8)
        bbox = [int(ymin), int(ymax), int(xmin), int(xmax)]
        focus = estimate_heatmap_focus(heatmap)
    else:
        conf_val = float(conf_map.mean().item())
        entropy_val = float(entropy_map.mean().item())
        status_label = "No Lesion Detected"
        risk = "Low Risk"
        bbox = []
        focus = "Diffuse / Background"

    confidence = round(max(0.01, min(0.99, conf_val)), 3)
    entropy = round(max(0.01, min(0.99, entropy_val)), 3)

    # 7. Render Base64 Images
    orig_b64 = array_to_base64_jpg(orig_rgb)
    clahe_b64 = array_to_base64_jpg(clahe_rgb)
    mask_overlay_rgb = create_mask_overlay(clahe_rgb, mask_np)
    mask_b64 = array_to_base64_jpg(mask_overlay_rgb)
    gradcam_overlay_rgb = create_gradcam_overlay(clahe_rgb, heatmap)
    gradcam_b64 = array_to_base64_jpg(gradcam_overlay_rgb)

    return {
        "status": "success",
        "quality": {
            "laplacian_variance": round(lap_var, 1),
            "is_blurred": is_blurred,
            "brightness_pct": round(brightness * 100, 1),
            "is_overexposed": is_overexposed,
            "is_underexposed": is_underexposed,
            "quality_label": quality,
            "clahe_applied": True
        },
        "segmentation": {
            "polyp_detected": is_polyp,
            "status_label": status_label,
            "lesion_area_px": lesion_area,
            "lesion_area_pct": lesion_pct,
            "confidence": confidence,
            "entropy": entropy,
            "risk_level": risk,
            "bounding_box": bbox,
            "model": "Attention U-Net (EXP-08 Verified Checkpoint)"
        },
        "explainability": {
            "method": "Segmentation Grad-CAM (Decoder up4)",
            "focus_region": focus,
            "attention_score": round(float(np.max(heatmap)), 2),
            "description": "Highlights image regions that contributed to the segmentation prediction."
        },
        "images": {
            "original": orig_b64,
            "clahe": clahe_b64,
            "mask": mask_b64,
            "gradcam": gradcam_b64
        },
        # Backwards compatibility keys for legacy references
        "classification": {
            "prediction": "Polyp" if is_polyp else "Normal",
            "confidence": confidence,
            "entropy": entropy,
            "risk_level": risk
        },
        "preprocessing": {
            "laplacian_variance": round(lap_var, 1),
            "is_blurred": is_blurred,
            "quality": quality,
            "clahe_applied": True
        }
    }


# ─────────────────────────────────────────────
# In-Memory Patient Session Store
# ─────────────────────────────────────────────
patient_sessions = {}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_frame():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    patient_id = request.form.get("patient_id", "PATIENT-001").strip() or "PATIENT-001"

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    result = analyze_image_with_model(pil_image)
    result["patient_id"] = patient_id

    # Store frame in patient session
    if patient_id not in patient_sessions:
        patient_sessions[patient_id] = {
            "patient_id": patient_id,
            "study_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "frames": []
        }

    frame_record = {
        "frame_id": f"frame_{len(patient_sessions[patient_id]['frames']) + 1:03d}",
        "filename": file.filename,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "polyp_detected": result["segmentation"]["polyp_detected"],
        "status_label": result["segmentation"]["status_label"],
        "lesion_area_px": result["segmentation"]["lesion_area_px"],
        "lesion_area_pct": result["segmentation"]["lesion_area_pct"],
        "confidence": result["segmentation"]["confidence"],
        "entropy": result["segmentation"]["entropy"],
        "bounding_box": result["segmentation"]["bounding_box"],
        "risk_level": result["segmentation"]["risk_level"],
        "quality_label": result["quality"]["quality_label"],
        "lap_var": result["quality"]["laplacian_variance"],
        "brightness_pct": result["quality"]["brightness_pct"],
        "focus": result["explainability"]["focus_region"],
        "images": {
            "original": result["images"]["original"],
            "mask": result["images"]["mask"],
            "gradcam": result["images"]["gradcam"]
        }
    }
    patient_sessions[patient_id]["frames"].append(frame_record)

    # Session stats
    all_frames = patient_sessions[patient_id]["frames"]
    polyp_count = sum(1 for f in all_frames if f["polyp_detected"])
    max_area = max((f["lesion_area_px"] for f in all_frames), default=0)

    result["session"] = {
        "patient_id": patient_id,
        "total_frames": len(all_frames),
        "polyp_frames": polyp_count,
        "clean_frames": len(all_frames) - polyp_count,
        "max_lesion_area_px": max_area,
        "recent_frames": [
            {
                "frame_id": f["frame_id"],
                "filename": f["filename"],
                "polyp_detected": f["polyp_detected"],
                "lesion_area_px": f["lesion_area_px"],
                "confidence": f["confidence"]
            }
            for f in all_frames[-6:]
        ]
    }

    return jsonify(result)


@app.route("/api/session/<patient_id>", methods=["GET"])
def get_session(patient_id):
    """Returns patient session stats and recent frames."""
    if patient_id not in patient_sessions:
        return jsonify({
            "patient_id": patient_id,
            "total_frames": 0,
            "polyp_frames": 0,
            "clean_frames": 0,
            "max_lesion_area_px": 0,
            "frames": []
        })
    s = patient_sessions[patient_id]
    frames = s["frames"]
    polyp_count = sum(1 for f in frames if f["polyp_detected"])
    return jsonify({
        "patient_id": patient_id,
        "study_date": s["study_date"],
        "total_frames": len(frames),
        "polyp_frames": polyp_count,
        "clean_frames": len(frames) - polyp_count,
        "max_lesion_area_px": max((f["lesion_area_px"] for f in frames), default=0),
        "frames": [
            {
                "frame_id": f["frame_id"],
                "filename": f["filename"],
                "polyp_detected": f["polyp_detected"],
                "lesion_area_px": f["lesion_area_px"],
                "confidence": f["confidence"]
            }
            for f in frames
        ]
    })


@app.route("/api/session/<patient_id>/reset", methods=["POST"])
def reset_session(patient_id):
    """Clears all analyzed frames for a patient."""
    if patient_id in patient_sessions:
        del patient_sessions[patient_id]
    return jsonify({"status": "reset", "patient_id": patient_id})


@app.route("/api/report/<patient_id>", methods=["GET"])
def generate_report(patient_id):
    """
    Renders an academic, print-ready, clean medical AI diagnostic report
    containing patient summary, frame-by-frame visual gallery, and statistics.
    """
    if patient_id not in patient_sessions or len(patient_sessions[patient_id]["frames"]) == 0:
        frames = []
        study_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    else:
        frames = patient_sessions[patient_id]["frames"]
        study_date = patient_sessions[patient_id]["study_date"]

    total = len(frames)
    polyp_frames = [f for f in frames if f["polyp_detected"]]
    clean_frames = [f for f in frames if not f["polyp_detected"]]
    polyp_rate = round(len(polyp_frames) / total * 100, 1) if total else 0.0
    avg_conf = round(sum(f["confidence"] for f in frames) / total * 100, 1) if total else 0.0
    max_area = max((f["lesion_area_px"] for f in frames), default=0)
    has_polyp = len(polyp_frames) > 0
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Banner message
    if total == 0:
        status_bg = "#f1f5f9"
        status_color = "#475569"
        status_border = "#cbd5e1"
        status_title = "No Frames Analyzed"
        status_desc = "No endoscopy frames have been analyzed for this patient session yet."
    elif has_polyp:
        status_bg = "#fef2f2"
        status_color = "#991b1b"
        status_border = "#fecaca"
        status_title = f"Lesions Detected in {len(polyp_frames)} of {total} Frames"
        status_desc = "Automated Attention U-Net identified positive polyp regions requiring clinical review."
    else:
        status_bg = "#f0fdf4"
        status_color = "#166534"
        status_border = "#bbf7d0"
        status_title = "No Polyp Lesions Detected"
        status_desc = "All analyzed endoscopy frames show clean mucosal tissue with no segmented lesions."

    # Frame-by-frame visual inspection blocks
    frame_blocks = ""
    for i, f in enumerate(frames):
        is_p = f["polyp_detected"]
        badge_style = "background:#fee2e2;color:#991b1b;border:1px solid #fecaca;" if is_p else "background:#dcfce7;color:#166534;border:1px solid #bbf7d0;"
        badge_text = "POLYP DETECTED" if is_p else "NO LESION"
        bbox_str = str(f["bounding_box"]) if f["bounding_box"] else "None"

        frame_blocks += f"""
        <div class="frame-card">
            <div class="frame-card-header">
                <div>
                    <span class="frame-idx">Frame #{i+1:02d}</span>
                    <strong style="margin-left:0.5rem;font-size:0.95rem;color:#1e293b;">{f['filename']}</strong>
                    <span style="margin-left:0.75rem;color:#64748b;font-size:0.8rem;">Recorded at {f['timestamp']}</span>
                </div>
                <span class="badge" style="{badge_style}">{badge_text}</span>
            </div>
            
            <div class="frame-visual-grid">
                <div class="img-box">
                    <img src="{f['images']['original']}" alt="Original Frame">
                    <div class="img-caption">1. Original Frame</div>
                </div>
                <div class="img-box">
                    <img src="{f['images']['mask']}" alt="Attention U-Net Mask">
                    <div class="img-caption">2. Attention U-Net Prediction</div>
                </div>
                <div class="img-box">
                    <img src="{f['images']['gradcam']}" alt="Segmentation Grad-CAM">
                    <div class="img-caption">3. Segmentation Grad-CAM</div>
                </div>
            </div>

            <div class="frame-meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Lesion Area</span>
                    <span class="meta-val">{f['lesion_area_px']:,} px² ({f['lesion_area_pct']}%)</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Confidence</span>
                    <span class="meta-val">{f['confidence'] * 100:.1f}%</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Predictive Entropy</span>
                    <span class="meta-val">{f['entropy']:.3f}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Bounding Box</span>
                    <span class="meta-val" style="font-family:monospace;font-size:0.82rem;">{bbox_str}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Image Quality</span>
                    <span class="meta-val">{f['quality_label']} (Lap: {f['lap_var']})</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Grad-CAM Focus</span>
                    <span class="meta-val">{f['focus']}</span>
                </div>
            </div>
        </div>
        """

    if not frame_blocks:
        frame_blocks = """
        <div style="text-align:center;padding:3rem 1rem;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;color:#64748b;">
            <p style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">No Frames Available</p>
            <p style="font-size:0.9rem;">Upload and analyze images in the Frame Inference tab to populate this diagnostic study.</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patient Diagnostic Report — {patient_id}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #f8fafc;
    color: #0f172a;
    padding: 2.5rem 1.5rem;
    line-height: 1.5;
  }}
  .container {{
    max-width: 1040px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 2.5rem;
    box-shadow: 0 4px 20px -4px rgba(0,0,0,0.05);
  }}
  
  /* Header */
  .report-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1.5rem;
  }}
  .title-area h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
  }}
  .title-area p {{
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 0.2rem;
  }}
  .action-area {{
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }}
  .print-btn {{
    background: #2563eb;
    color: #ffffff;
    border: none;
    padding: 0.5rem 1.1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }}
  .print-btn:hover {{ background: #1d4ed8; }}

  /* Metadata Strip */
  .meta-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.9rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
  }}
  .meta-strip span {{ color: #64748b; }}
  .meta-strip strong {{ color: #0f172a; margin-left: 0.35rem; }}

  /* Diagnostic Status Banner */
  .status-banner {{
    background: {status_bg};
    border: 1px solid {status_border};
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.75rem;
  }}
  .status-banner h3 {{
    color: {status_color};
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
  }}
  .status-banner p {{
    color: {status_color};
    opacity: 0.9;
    font-size: 0.88rem;
  }}

  /* Stats Grid */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 2rem;
  }}
  .stat-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 0.8rem;
    text-align: center;
  }}
  .stat-val {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
  }}
  .stat-lbl {{
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
  }}

  /* Section Titles */
  .sec-heading {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e2e8f0;
  }}

  /* Frame Card */
  .frame-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
  }}
  .frame-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #f1f5f9;
  }}
  .frame-idx {{
    background: #e2e8f0;
    color: #334155;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
  }}
  .badge {{
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }}

  /* Visual 3-Grid */
  .frame-visual-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1rem;
  }}
  .img-box {{
    background: #000000;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
    text-align: center;
  }}
  .img-box img {{
    width: 100%;
    aspect-ratio: 1/1;
    object-fit: cover;
    display: block;
  }}
  .img-caption {{
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    padding: 0.35rem 0.5rem;
  }}

  /* Frame Meta Grid */
  .frame-meta-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem 1rem;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
  }}
  .meta-item {{ display: flex; justify-content: space-between; }}
  .meta-label {{ color: #64748b; }}
  .meta-val {{ font-weight: 600; color: #1e293b; }}

  /* Disclaimer Box */
  .disclaimer-box {{
    margin-top: 2rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    padding: 0.9rem 1.25rem;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.45;
  }}

  /* Footer */
  .report-footer {{
    margin-top: 1.75rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #94a3b8;
  }}

  /* Print specific */
  @media print {{
    body {{ background: #ffffff; padding: 0; }}
    .container {{ border: none; box-shadow: none; padding: 0; max-width: 100%; }}
    .print-btn {{ display: none; }}
    .frame-card {{ break-inside: avoid; border: 1px solid #cbd5e1; }}
  }}
</style>
</head>
<body>

<div class="container">
  <!-- Header -->
  <div class="report-header">
    <div class="title-area">
      <h1>Endoscopic Lesion Diagnostic Report</h1>
      <p>Attention-Guided U-Net Segmentation & Explainability Study</p>
    </div>
    <div class="action-area">
      <button class="print-btn" onclick="window.print()">🖨 Print / Save PDF</button>
    </div>
  </div>

  <!-- Metadata -->
  <div class="meta-strip">
    <div><span>Patient ID:</span> <strong>{patient_id}</strong></div>
    <div><span>Study Session:</span> <strong>{study_date}</strong></div>
    <div><span>Frames Evaluated:</span> <strong>{total}</strong></div>
    <div><span>Generated:</span> <strong>{generated_at}</strong></div>
  </div>

  <!-- Status Banner -->
  <div class="status-banner">
    <h3>{status_title}</h3>
    <p>{status_desc}</p>
  </div>

  <!-- Summary Statistics -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-val">{total}</div>
      <div class="stat-lbl">Total Frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#dc2626;">{len(polyp_frames)}</div>
      <div class="stat-lbl">Polyp Frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#16a34a;">{len(clean_frames)}</div>
      <div class="stat-lbl">Clean Frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#2563eb;">{avg_conf}%</div>
      <div class="stat-lbl">Avg Confidence</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color:#7c3aed;">{max_area:,}</div>
      <div class="stat-lbl">Max Area (px²)</div>
    </div>
  </div>

  <!-- Frame-by-Frame Inspection Gallery -->
  <h2 class="sec-heading">Frame-by-Frame Diagnostic Inspection</h2>
  {frame_blocks}

  <!-- Research Disclaimer -->
  <div class="disclaimer-box">
    <strong>Research Prototype Disclaimer:</strong> This report is automatically generated by an academic computer-aided detection prototype implementing Attention-Guided U-Net and Segmentation Grad-CAM. Results are intended strictly for research and investigative evaluation. This report is not a clinical diagnosis and must not replace professional endoscopic evaluation by a licensed gastroenterologist.
  </div>

  <!-- Footer -->
  <div class="report-footer">
    <span>MIP Project · Attention-Guided U-Net Colonoscopy CAD</span>
    <span>In-Domain Dice: 0.8007 · Cross-Domain CVC Dice: 0.6568</span>
  </div>
</div>

</body>
</html>
"""
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html"
    return resp


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=True)
