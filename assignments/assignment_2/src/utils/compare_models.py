"""
Comparison Script: Baseline vs Fine-Tuned Model
Author: Usman Amjad
Assignment 2 - SemEval 2026 Task 4

This script compares the performance of the baseline SentenceTransformer model
with the fine-tuned model on the dev set.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Import evaluation function from main script
import sys
sys.path.append(str(Path(__file__).parents[1]))
from model_4_finetuned_sbert import evaluate_model

# Setup paths
ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "assignments" / "assignment_2" / "output" / "usman_amjad"


def main():
    """
    Compare baseline and fine-tuned models.
    """
    print("\n" + "="*60)
    print("Model Comparison: Baseline vs Fine-Tuned")
    print("Author: Usman Amjad")
    print("="*60 + "\n")
    
    dev_data = DATA_DIR / "raw" / "dev_track_a.jsonl"
    finetuned_path = OUTPUT_DIR / "finetuned_model"
    
    # Check if fine-tuned model exists
    if not finetuned_path.exists():
        print("ERROR: Fine-tuned model not found!")
        print(f"Expected location: {finetuned_path}")
        print("\nPlease run usman_finetune_embeddings.py first to train the model.")
        return
    
    # Evaluate baseline model
    print("\n[1/2] Evaluating Baseline Model")
    print("-" * 60)
    print("Loading baseline model: all-MiniLM-L6-v2 (pre-trained, no fine-tuning)")
    baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
    baseline_acc = evaluate_model(baseline_model, dev_data)
    
    # Evaluate fine-tuned model
    print("\n[2/2] Evaluating Fine-Tuned Model")
    print("-" * 60)
    print(f"Loading fine-tuned model from: {finetuned_path}")
    finetuned_model = SentenceTransformer(str(finetuned_path))
    finetuned_acc = evaluate_model(finetuned_model, dev_data)
    
    # Calculate improvement
    absolute_improvement = finetuned_acc - baseline_acc
    relative_improvement = (absolute_improvement / baseline_acc) * 100
    
    # Display comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline Model:       {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"Fine-Tuned Model:     {finetuned_acc:.4f} ({finetuned_acc*100:.2f}%)")
    print("-" * 60)
    print(f"Absolute Improvement: {absolute_improvement:+.4f} ({absolute_improvement*100:+.2f} percentage points)")
    print(f"Relative Improvement: {relative_improvement:+.2f}%")
    print("="*60 + "\n")
    
    # Save comparison results
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "model": "all-MiniLM-L6-v2",
            "accuracy": baseline_acc,
            "description": "Pre-trained SentenceTransformer without fine-tuning"
        },
        "finetuned": {
            "model": "all-MiniLM-L6-v2 (fine-tuned)",
            "accuracy": finetuned_acc,
            "description": "Fine-tuned on synthetic contrastive learning data"
        },
        "improvement": {
            "absolute": absolute_improvement,
            "relative_percent": relative_improvement
        }
    }
    
    comparison_file = OUTPUT_DIR / "comparison_results.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Comparison results saved to: {comparison_file}\n")
    
    # Interpretation
    print("INTERPRETATION:")
    if relative_improvement > 5:
        print("✓ Significant improvement! Fine-tuning was successful.")
    elif relative_improvement > 0:
        print("✓ Moderate improvement. Fine-tuning helped but gains are modest.")
    else:
        print("⚠ No improvement or degradation. Consider:")
        print("  - Increasing training epochs")
        print("  - Adjusting batch size")
        print("  - Checking data quality")
        print("  - Trying different base models")
    print()


if __name__ == "__main__":
    main()
