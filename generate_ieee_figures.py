import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.makedirs("ieee_report/figures", exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# 1. Proposed Workflow Diagram (IEEE Conference Two-Column Width)
# ==============================================================================
def generate_proposed_workflow():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Stages definition
    boxes = [
        {"x": 0.04, "y": 0.55, "w": 0.15, "h": 0.32, "title": "Input Colonoscopy", "sub": "RGB Image\n256 × 256 pixels\n(Kvasir / CVC)", "col": "#EBF8FF", "border": "#3182CE"},
        {"x": 0.23, "y": 0.55, "w": 0.16, "h": 0.32, "title": "Preprocessing & QC", "sub": "Laplacian Blur Check\nExposure Filtering\nCIE LAB CLAHE (L*)", "col": "#E6FFFA", "border": "#319795"},
        {"x": 0.43, "y": 0.55, "w": 0.18, "h": 0.32, "title": "Attention U-Net", "sub": "Encoder-Decoder\nAdditive Attention Gates\n7.85M Parameters", "col": "#FAF5FF", "border": "#805AD5"},
        {"x": 0.65, "y": 0.55, "w": 0.15, "h": 0.32, "title": "Segmentation", "sub": "Binary Mask (τ=0.5)\nBoundary Contour\n10.29 FPS Throughput", "col": "#F0FFF4", "border": "#38A169"},
        {"x": 0.84, "y": 0.55, "w": 0.14, "h": 0.32, "title": "Explainability", "sub": "Decoder Grad-CAM\nActivation Heatmap\nLocalization Audit", "col": "#FFF5F5", "border": "#E53E3E"}
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b['x'], b['y']), b['w'], b['h'],
                                     boxstyle="round,pad=0.015,rounding_size=0.02",
                                     linewidth=1.2, edgecolor=b['border'], facecolor=b['col'])
        ax.add_patch(rect)
        ax.text(b['x'] + b['w']/2, b['y'] + b['h'] - 0.06, b['title'],
                ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A202C')
        ax.text(b['x'] + b['w']/2, b['y'] + 0.11, b['sub'],
                ha='center', va='center', fontsize=7.5, color='#4A5568', linespacing=1.2)

    # Connecting arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i]['x'] + boxes[i]['w']
        x_end = boxes[i+1]['x']
        y_pos = boxes[i]['y'] + boxes[i]['h']/2
        ax.annotate('', xy=(x_end, y_pos), xytext=(x_start, y_pos),
                    arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5, mutation_scale=14))

    # Bottom Stage Details / Pipeline Metadata
    meta_box = patches.FancyBboxPatch((0.04, 0.08), 0.94, 0.36,
                                     boxstyle="round,pad=0.015,rounding_size=0.02",
                                     linewidth=1.0, edgecolor="#CBD5E0", facecolor="#F7FAFC")
    ax.add_patch(meta_box)

    ax.text(0.06, 0.36, "Operational Telemetry & Protocol Specifications:",
            fontsize=8.5, fontweight='bold', color='#2D3748')

    telemetry = [
        ("Phase 1: Input Filtering", "• Bilinear image resizing (256×256); Nearest-neighbor mask interpolation.\n• Motion blur threshold: Variance of Laplacian < 80.0 flags corrupted frames.\n• Contrast normalization: CLAHE on luminance (L*) with clip limit 2.0, tile 8×8."),
        ("Phase 2: Deep Inference", "• Additive soft attention gates filter coarse skip connections before concatenation.\n• Loss formulation: Compound Binary Cross-Entropy + Dice Loss (1.0 + 1.0).\n• Inference throughput: 10.29 FPS (97.1 ms latency per frame on standard hardware)."),
        ("Phase 3: Interpretability", "• Target layer: Final decoder block ('model.up4') preserving high spatial resolution.\n• Visual alignment: Overlays localization heatmaps for clinician verification.\n• Quantitative measure: Mean localization Dice of 0.3280 over ground-truth lesions.")
    ]

    col_xs = [0.06, 0.38, 0.70]
    for idx, (title, desc) in enumerate(telemetry):
        ax.text(col_xs[idx], 0.28, title, fontsize=8, fontweight='bold', color='#3182CE' if idx==0 else ('#805AD5' if idx==1 else '#E53E3E'))
        ax.text(col_xs[idx], 0.18, desc, fontsize=6.8, color='#4A5568', va='top', linespacing=1.2)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.95)
    plt.tight_layout()
    plt.savefig("ieee_report/figures/proposed_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated proposed_workflow.png")

# ==============================================================================
# 2. Attention U-Net Architecture Diagram
# ==============================================================================
def generate_attention_unet_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.axis('off')

    # Draw encoder blocks
    enc_x = [0.08, 0.18, 0.28, 0.38]
    enc_y = [0.72, 0.58, 0.44, 0.30]
    enc_channels = ["64", "128", "256", "512"]
    enc_dims = ["256×256", "128×128", "64×64", "32×32"]

    for i in range(4):
        rect = patches.FancyBboxPatch((enc_x[i]-0.04, enc_y[i]-0.05), 0.08, 0.10,
                                     boxstyle="round,pad=0.01", edgecolor="#2B6CB0", facecolor="#EBF8FF", lw=1.2)
        ax.add_patch(rect)
        ax.text(enc_x[i], enc_y[i]+0.015, f"Enc {i+1}\n({enc_channels[i]})", ha='center', va='center', fontsize=8, fontweight='bold', color='#1A365D')
        ax.text(enc_x[i], enc_y[i]-0.035, enc_dims[i], ha='center', va='center', fontsize=6.5, color='#4A5568')

        if i < 3:
            ax.annotate('', xy=(enc_x[i+1]-0.04, enc_y[i+1]+0.05), xytext=(enc_x[i]+0.04, enc_y[i]-0.05),
                        arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.2, mutation_scale=10))
            ax.text((enc_x[i]+enc_x[i+1])/2, (enc_y[i]+enc_y[i+1])/2, "MaxPool 2×2", fontsize=6, color="#C53030", ha='center', va='bottom', rotation=-30)

    # Bottleneck
    btn_x, btn_y = 0.48, 0.15
    rect = patches.FancyBboxPatch((btn_x-0.05, btn_y-0.05), 0.10, 0.10,
                                 boxstyle="round,pad=0.01", edgecolor="#6B46C1", facecolor="#FAF5FF", lw=1.4)
    ax.add_patch(rect)
    ax.text(btn_x, btn_y+0.015, "Bottleneck\n(1024)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#44337A')
    ax.text(btn_x, btn_y-0.035, "16×16", ha='center', va='center', fontsize=6.5, color='#4A5568')

    ax.annotate('', xy=(btn_x-0.05, btn_y+0.05), xytext=(enc_x[3]+0.04, enc_y[3]-0.05),
                arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.2, mutation_scale=10))

    # Decoder blocks
    dec_x = [0.58, 0.68, 0.78, 0.88]
    dec_y = [0.30, 0.44, 0.58, 0.72]
    dec_channels = ["512", "256", "128", "64"]
    dec_dims = ["32×32", "64×64", "128×128", "256×256"]

    ax.annotate('', xy=(dec_x[0]-0.04, dec_y[0]-0.05), xytext=(btn_x+0.05, btn_y+0.05),
                arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.2, mutation_scale=10))
    ax.text((btn_x+dec_x[0])/2, (btn_y+dec_y[0])/2, "UpConv", fontsize=6, color="#2B6CB0", ha='center', va='top', rotation=35)

    for i in range(4):
        rect = patches.FancyBboxPatch((dec_x[i]-0.04, dec_y[i]-0.05), 0.08, 0.10,
                                     boxstyle="round,pad=0.01", edgecolor="#234E52", facecolor="#E6FFFA", lw=1.2)
        ax.add_patch(rect)
        ax.text(dec_x[i], dec_y[i]+0.015, f"Dec {i+1}\n({dec_channels[i]})", ha='center', va='center', fontsize=8, fontweight='bold', color='#234E52')
        ax.text(dec_x[i], dec_y[i]-0.035, dec_dims[i], ha='center', va='center', fontsize=6.5, color='#4A5568')

        if i < 3:
            ax.annotate('', xy=(dec_x[i+1]-0.04, dec_y[i+1]-0.05), xytext=(dec_x[i]+0.04, dec_y[i]+0.05),
                        arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.2, mutation_scale=10))
            ax.text((dec_x[i]+dec_x[i+1])/2, (dec_y[i]+dec_y[i+1])/2, "UpConv 2×2", fontsize=6, color="#2B6CB0", ha='center', va='top', rotation=35)

    # Attention Gates across skips
    ag_y = [0.72, 0.58, 0.44, 0.30]
    for i in range(4):
        # Attention Gate Box
        ag_x = (enc_x[3-i] + dec_x[i]) / 2
        ag_box = patches.FancyBboxPatch((ag_x-0.035, ag_y[3-i]-0.035), 0.07, 0.07,
                                       boxstyle="round,pad=0.008", edgecolor="#DD6B20", facecolor="#FFFAF0", lw=1.2)
        ax.add_patch(ag_box)
        ax.text(ag_x, ag_y[3-i], f"AG {4-i}", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#C05621')

        # Skip from encoder to AG
        ax.annotate('', xy=(ag_x-0.035, ag_y[3-i]), xytext=(enc_x[3-i]+0.04, enc_y[3-i]),
                    arrowprops=dict(arrowstyle="->", color="#DD6B20", lw=1.0, linestyle="--", mutation_scale=8))

        # Gating signal from decoder below to AG
        gate_src_y = btn_y if i == 0 else dec_y[i-1]
        gate_src_x = btn_x if i == 0 else dec_x[i-1]
        ax.annotate('', xy=(ag_x, ag_y[3-i]-0.035), xytext=(gate_src_x, gate_src_y+0.05),
                    arrowprops=dict(arrowstyle="->", color="#805AD5", lw=1.0, linestyle=":", mutation_scale=8))

        # Gated output from AG to decoder
        ax.annotate('', xy=(dec_x[i]-0.04, dec_y[i]), xytext=(ag_x+0.035, ag_y[3-i]),
                    arrowprops=dict(arrowstyle="->", color="#DD6B20", lw=1.0, mutation_scale=8))

    # Attention Gate Mathematics Inset (Bottom Left)
    inset_box = patches.FancyBboxPatch((0.04, 0.04), 0.38, 0.17,
                                      boxstyle="round,pad=0.01", edgecolor="#DD6B20", facecolor="#FFFDF7", lw=1.0)
    ax.add_patch(inset_box)
    ax.text(0.06, 0.18, "Additive Soft Attention Gate Formulation:", fontsize=7.5, fontweight='bold', color='#C05621')
    ax.text(0.06, 0.13, r"$\alpha = \sigma\left(\psi^T(\text{ReLU}(W_g g + W_x x + b_g))\right)$", fontsize=7.5, color='#1A202C')
    ax.text(0.06, 0.07, r"$\hat{x} = x \cdot \alpha, \quad \alpha \in [0, 1]$ (Suppresses background)", fontsize=7.0, color='#4A5568')

    # Final Output
    ax.annotate('', xy=(0.98, dec_y[3]), xytext=(dec_x[3]+0.04, dec_y[3]),
                arrowprops=dict(arrowstyle="->", color="#2D3748", lw=1.5, mutation_scale=12))
    ax.text(0.98, dec_y[3]+0.06, "1×1 Conv\n+ Sigmoid", ha='center', va='center', fontsize=7, color='#2D3748')
    ax.text(0.98, dec_y[3]-0.06, "Mask\n(256×256)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#2B6CB0')

    # Legend / Parameter info
    param_box = patches.FancyBboxPatch((0.64, 0.04), 0.32, 0.17,
                                      boxstyle="round,pad=0.01", edgecolor="#CBD5E0", facecolor="#F7FAFC", lw=1.0)
    ax.add_patch(param_box)
    ax.text(0.66, 0.18, "Model Parameter Audit:", fontsize=7.5, fontweight='bold', color='#2D3748')
    ax.text(0.66, 0.13, "• Attention U-Net: 7,851,773 params", fontsize=7, color='#2B6CB0')
    ax.text(0.66, 0.09, "• Standard U-Net:   7,763,041 params", fontsize=7, color='#718096')
    ax.text(0.66, 0.05, "• Overhead: +88,732 params (+1.14%)", fontsize=7, fontweight='bold', color='#234E52')

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 0.88)
    plt.tight_layout()
    plt.savefig("ieee_report/figures/attention_unet_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated attention_unet_architecture.png")

# ==============================================================================
# 3. Model Comparison Plot (Attention U-Net vs Standard U-Net)
# ==============================================================================
def generate_model_comparison():
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'Specificity']
    attn_means = [0.8093, 0.7298, 0.8229, 0.8769, 0.9650]
    attn_stds  = [0.2325, 0.2603, 0.2401, 0.2127, 0.0674]
    std_means  = [0.8001, 0.7210, 0.8471, 0.8413, 0.9716]
    std_stds   = [0.2450, 0.2689, 0.2265, 0.2462, 0.0518]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, attn_means, width, yerr=attn_stds, label='Attention U-Net (7.85M)',
                   color='#2B6CB0', edgecolor='#1A365D', capsize=3, alpha=0.9, error_kw=dict(lw=0.8, capthick=0.8))
    bars2 = ax.bar(x + width/2, std_means, width, yerr=std_stds, label='Standard U-Net (7.76M)',
                   color='#718096', edgecolor='#2D3748', capsize=3, alpha=0.9, error_kw=dict(lw=0.8, capthick=0.8))

    ax.set_ylabel('Metric Score (Mean ± Std)', fontsize=9, fontweight='bold', color='#1A202C')
    ax.set_title('Controlled In-Domain Architectural Comparison (Kvasir-SEG, 99 Images)', fontsize=9.5, fontweight='bold', color='#1A202C', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.5, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Highlight gains
    for i in range(len(metrics)):
        diff = attn_means[i] - std_means[i]
        diff_str = f"{diff:+.2%}" if abs(diff) >= 0.005 else f"{diff:+.4f}"
        color = '#2B6CB0' if diff > 0 else '#C53030'
        ax.text(x[i] - width/2, attn_means[i] + attn_stds[i] + 0.03, f"{attn_means[i]:.4f}", ha='center', va='bottom', fontsize=6.5, color='#1A365D', fontweight='bold')
        ax.text(x[i] + width/2, std_means[i] + std_stds[i] + 0.03, f"{std_means[i]:.4f}", ha='center', va='bottom', fontsize=6.5, color='#2D3748')

    plt.tight_layout()
    plt.savefig("ieee_report/figures/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated model_comparison.png")

# ==============================================================================
# 4. External Evaluation Progression Plot
# ==============================================================================
def generate_external_evaluation():
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    stages = ['Kvasir In-Domain\n(100 Frames)', 'CVC Baseline\n(Zero-Shot)', 'CVC Inference\nCLAHE', 'CVC Retrained\nCLAHE (EXP-08)']
    dice_means = [0.8120, 0.4516, 0.5702, 0.6568]
    dice_stds  = [0.2282, 0.3196, 0.3020, 0.3297]
    iou_means  = [0.7307, 0.3495, 0.4594, 0.5676]
    iou_stds   = [0.2506, 0.2849, 0.2916, 0.3248]

    x = np.arange(len(stages))
    width = 0.35

    rects1 = ax.bar(x - width/2, dice_means, width, yerr=dice_stds, label='Dice Similarity Coefficient',
                    color='#319795', edgecolor='#234E52', capsize=3, alpha=0.9, error_kw=dict(lw=0.8, capthick=0.8))
    rects2 = ax.bar(x + width/2, iou_means, width, yerr=iou_stds, label='Intersection-over-Union (IoU)',
                    color='#DD6B20', edgecolor='#C05621', capsize=3, alpha=0.9, error_kw=dict(lw=0.8, capthick=0.8))

    ax.set_ylabel('Metric Score (Mean ± Std)', fontsize=9, fontweight='bold', color='#1A202C')
    ax.set_title('Cross-Dataset Generalization & Adaptation Progression (CVC-ClinicDB, 612 Frames)', fontsize=9.5, fontweight='bold', color='#1A202C', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Annotate values
    for i in range(len(stages)):
        ax.text(x[i] - width/2, dice_means[i] + dice_stds[i] + 0.03, f"{dice_means[i]:.4f}", ha='center', va='bottom', fontsize=6.8, color='#234E52', fontweight='bold')
        ax.text(x[i] + width/2, iou_means[i] + iou_stds[i] + 0.03, f"{iou_means[i]:.4f}", ha='center', va='bottom', fontsize=6.8, color='#C05621', fontweight='bold')

    # Draw progression annotation
    ax.annotate('+20.52% Dice Gain\n(+45.4% Relative)',
                xy=(x[3]-width/2, dice_means[3]), xytext=(x[1]-width/2 + 0.5, 0.95),
                arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.2, connectionstyle="arc3,rad=-0.2"),
                fontsize=7.5, fontweight='bold', color="#C53030", bbox=dict(boxstyle="round,pad=0.2", fc="#FFF5F5", ec="#E53E3E", lw=0.8))

    plt.tight_layout()
    plt.savefig("ieee_report/figures/external_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated external_evaluation.png")

# ==============================================================================
# 5. Threshold Analysis Curves
# ==============================================================================
def generate_threshold_analysis():
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    tau = [0.1, 0.2, 0.3, 0.4, 0.5]
    dice = [0.4086, 0.4344, 0.4431, 0.4490, 0.4516]
    iou = [0.2980, 0.3252, 0.3358, 0.3444, 0.3495]
    prec = [0.3715, 0.4495, 0.4907, 0.5226, 0.5656]
    rec = [0.7776, 0.7251, 0.6930, 0.6536, 0.6124]

    ax.plot(tau, dice, 'o-', color='#319795', lw=1.8, ms=5, label='Dice Similarity (Peak: 0.4516 @ 0.5)')
    ax.plot(tau, iou, 's-', color='#DD6B20', lw=1.8, ms=5, label='IoU / Jaccard (Peak: 0.3495 @ 0.5)')
    ax.plot(tau, prec, '^-', color='#2B6CB0', lw=1.8, ms=5, label='Precision (Rises 0.3715 → 0.5656)')
    ax.plot(tau, rec, 'v--', color='#E53E3E', lw=1.8, ms=5, label='Recall (Drops 0.7776 → 0.6124)')

    ax.set_xlabel('Decision Binarization Threshold (τ)', fontsize=9, fontweight='bold', color='#1A202C')
    ax.set_ylabel('Evaluation Metric Score', fontsize=9, fontweight='bold', color='#1A202C')
    ax.set_title('Diagnostic Threshold Sensitivity Sweep on CVC-ClinicDB (612 Frames)', fontsize=9.5, fontweight='bold', color='#1A202C', pad=10)
    ax.set_xticks(tau)
    ax.set_ylim(0.20, 0.85)
    ax.grid(True, linestyle='--', alpha=0.4, color='#CBD5E0')
    ax.legend(loc='center left', fontsize=8, framealpha=0.9)

    ax.axvline(0.5, color='#319795', linestyle=':', lw=1.2)
    ax.text(0.5, 0.22, 'τ = 0.5 (Strictly Optimal)', ha='right', va='bottom', fontsize=7.5, fontweight='bold', color='#234E52', rotation=90)

    plt.tight_layout()
    plt.savefig("ieee_report/figures/threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated threshold_analysis.png")

# ==============================================================================
# 6. Qualitative Results (Two-Case Composite with Clean Scientific Labels)
# ==============================================================================
def generate_qualitative_results():
    # Load genuine viz assets
    viz0 = Image.open('results/review_1/visualizations/viz_00_cju2spdagu1l50835da1f46fr.png')
    viz2 = Image.open('results/review_1/visualizations/viz_02_cju35k2fr3vc50988c85qkrwg.png')

    w, h = viz0.size
    pw = w // 5 # 645 px per panel

    # Crop first 4 panels from each (Original, Ground Truth, Pred Mask, Boundary Overlay)
    p0 = [viz0.crop((i*pw, 0, (i+1)*pw, h)) for i in range(4)]
    p2 = [viz2.crop((i*pw, 0, (i+1)*pw, h)) for i in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('white')

    titles = ['(a) Input Colonoscopy', '(b) Ground Truth Mask', '(c) Attention U-Net Pred.', '(d) Segmentation Contour']

    for j in range(4):
        axes[0, j].imshow(p0[j])
        axes[0, j].axis('off')
        axes[0, j].set_title(titles[j], fontsize=8.5, fontweight='bold', color='#1A202C', pad=5)

        axes[1, j].imshow(p2[j])
        axes[1, j].axis('off')

    # Row labels
    axes[0, 0].text(-0.08, 0.5, 'Case 1:\nSessile Lesion\n(High Overlap)',
                    transform=axes[0, 0].transAxes, va='center', ha='right', fontsize=8, fontweight='bold', color='#2B6CB0')
    axes[1, 0].text(-0.08, 0.5, 'Case 2:\nIrregular Mucosa\n(Boundary Detail)',
                    transform=axes[1, 0].transAxes, va='center', ha='right', fontsize=8, fontweight='bold', color='#234E52')

    plt.tight_layout()
    plt.savefig("ieee_report/figures/qualitative_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated qualitative_results.png")

# ==============================================================================
# 7. CLAHE Comparison Figure
# ==============================================================================
def generate_clahe_comparison():
    im = Image.open('results/cvc_model_comparison/comparison_id_3.png')
    # Save directly with crisp high resolution
    im.save("ieee_report/figures/clahe_comparison.png", dpi=(300, 300))
    print("Generated clahe_comparison.png")

# ==============================================================================
# 8. Grad-CAM Visualization Figure
# ==============================================================================
def generate_gradcam_visualization():
    viz = Image.open('results/review_1/visualizations/viz_00_cju2spdagu1l50835da1f46fr.png')
    w, h = viz.size
    pw = w // 5

    panels = [viz.crop((i*pw, 0, (i+1)*pw, h)) for i in range(5)]
    titles = [
        '(a) Endoscopic Frame',
        '(b) Ground Truth Lesion',
        '(c) Attention U-Net Mask',
        '(d) Boundary Overlay',
        '(e) Decoder Grad-CAM (up4)'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), dpi=300)
    fig.patch.set_facecolor('white')

    for i in range(5):
        axes[i].imshow(panels[i])
        axes[i].axis('off')
        axes[i].set_title(titles[i], fontsize=8, fontweight='bold', color='#1A202C', pad=5)

    plt.tight_layout()
    plt.savefig("ieee_report/figures/gradcam_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated gradcam_visualization.png")

if __name__ == '__main__':
    generate_proposed_workflow()
    generate_attention_unet_architecture()
    generate_model_comparison()
    generate_external_evaluation()
    generate_threshold_analysis()
    generate_qualitative_results()
    generate_clahe_comparison()
    generate_gradcam_visualization()
    print("All 8 IEEE figures successfully generated!")
