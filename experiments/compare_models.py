import os
import json
from typing import Dict

def load_metrics(path: str) -> Dict[str, float]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data["summary"]

def main():
    baseline_path = os.path.join("results", "review_1", "metrics.json")
    dropout_path = os.path.join("results", "review_1_dropout", "metrics.json")
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline metrics not found at {baseline_path}")
    if not os.path.exists(dropout_path):
        raise FileNotFoundError(f"Dropout metrics not found at {dropout_path}")
        
    baseline_metrics = load_metrics(baseline_path)
    dropout_metrics = load_metrics(dropout_path)
    
    metrics = ["dice", "iou", "precision", "recall", "specificity"]
    
    # Generate markdown table
    md_content = "# Model Comparison: Baseline vs Dropout (0.2)\n\n"
    md_content += "| Metric | Baseline (Mean ± Std) | Dropout 0.2 (Mean ± Std) | Difference (Mean) |\n"
    md_content += "|---|---|---|---|\n"
    
    for metric in metrics:
        b_mean = baseline_metrics[f"mean_{metric}"]
        b_std = baseline_metrics[f"std_{metric}"]
        d_mean = dropout_metrics[f"mean_{metric}"]
        d_std = dropout_metrics[f"std_{metric}"]
        
        diff = d_mean - b_mean
        diff_str = f"{diff:+.4f}"
        
        md_content += f"| **{metric.capitalize()}** | {b_mean:.4f} ± {b_std:.4f} | {d_mean:.4f} ± {d_std:.4f} | {diff_str} |\n"
        
    # Save markdown
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_review_1_vs_dropout.md")
    with open(out_path, 'w') as f:
        f.write(md_content)
        
    print(f"Comparison saved to {out_path}")
    print("\n" + md_content)

if __name__ == "__main__":
    main()
