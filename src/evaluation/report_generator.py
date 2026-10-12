import json
import datetime
import os
from typing import List, Dict, Any

class DiagnosticReportGenerator:
    """
    Generates a patient-level HTML diagnostic report based on aggregated frame data.
    """
    def __init__(self, patient_id: str, study_date: str = None):
        self.patient_id = patient_id
        self.study_date = study_date or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.frame_data: List[Dict[str, Any]] = []
        
    def add_frame_analysis(self, frame_id: str, classification: str, confidence: float, lesion_area_px: int = 0):
        """
        Adds a single frame's analysis results to the patient's record.
        """
        self.frame_data.append({
            "frame_id": frame_id,
            "classification": classification,
            "confidence": confidence,
            "lesion_area_px": lesion_area_px
        })
        
    def aggregate_patient_results(self) -> Dict[str, Any]:
        """
        Aggregates frame-level data into patient-level insights.
        """
        total_frames = len(self.frame_data)
        if total_frames == 0:
            return {}
            
        polyp_frames = [f for f in self.frame_data if f["classification"].lower() == "polyp"]
        num_polyp_frames = len(polyp_frames)
        
        # Patient is flagged if any frame shows a highly confident polyp
        # (Alternatively, could require a minimum sequence length)
        flagged = num_polyp_frames > 0
        
        avg_confidence = sum(f["confidence"] for f in self.frame_data) / total_frames
        max_lesion_area = max(f["lesion_area_px"] for f in self.frame_data) if total_frames > 0 else 0
        
        return {
            "total_frames": total_frames,
            "polyp_frames": num_polyp_frames,
            "normal_frames": total_frames - num_polyp_frames,
            "is_flagged": flagged,
            "average_confidence": round(avg_confidence, 4),
            "max_lesion_area_px": max_lesion_area
        }

    def generate_html_report(self, output_path: str):
        """
        Generates a premium HTML report for the patient.
        """
        summary = self.aggregate_patient_results()
        
        status_color = "#ef4444" if summary.get("is_flagged") else "#10b981"
        status_text = "ACTION REQUIRED: POLYPS DETECTED" if summary.get("is_flagged") else "NORMAL: NO POLYPS DETECTED"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diagnostic Report - {self.patient_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6; margin: 40px auto; max-width: 800px; background: #f9fafb; }}
                .card {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; color: #111; }}
                .status-banner {{ background: {status_color}; color: white; padding: 15px; border-radius: 6px; font-weight: bold; text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f3f4f6; font-weight: 600; }}
                .highlight {{ font-weight: bold; color: {status_color}; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Endoscopic Lesion Diagnostic Report</h1>
                <p><strong>Patient ID:</strong> {self.patient_id}</p>
                <p><strong>Study Date:</strong> {self.study_date}</p>
                
                <div class="status-banner">
                    {status_text}
                </div>
                
                <h2>Summary Statistics</h2>
                <ul>
                    <li>Total Frames Analyzed: {summary.get('total_frames', 0)}</li>
                    <li>Frames with Polyps: {summary.get('polyp_frames', 0)}</li>
                    <li>Average Model Confidence: {summary.get('average_confidence', 0)}</li>
                    <li>Max Lesion Area Detected: {summary.get('max_lesion_area_px', 0)} px</li>
                </ul>
                
                <h2>Frame-Level Breakdown</h2>
                <table>
                    <tr>
                        <th>Frame ID</th>
                        <th>Classification</th>
                        <th>Confidence</th>
                        <th>Lesion Area (px)</th>
                    </tr>
                    {"".join(
                        f"<tr><td>{f['frame_id']}</td>"
                        f"<td class='{'highlight' if f['classification'].lower()=='polyp' else ''}'>{f['classification']}</td>"
                        f"<td>{f['confidence']:.2f}</td>"
                        f"<td>{f['lesion_area_px']}</td></tr>"
                        for f in self.frame_data
                    )}
                </table>
            </div>
        </body>
        </html>
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path
