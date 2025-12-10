"""
Generate comparison tables for Assignment 3 report.
Author: Muhammad Musaab ul Haq
"""

import json
import csv
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# TABLE 1: MODEL COMPARISON (All Assignments)
# ============================================================================

def generate_model_comparison():
    """Generate comprehensive model comparison table."""
    
    # Try to load actual hybrid results
    hybrid_summary = SCRIPT_DIR.parent / "output" / "musaab_hybrid" / "hybrid_summary_t0.1.json"
    hybrid_acc = 0.605
    hybrid_fast = 0.45
    hybrid_slow = 0.55
    hybrid_fast_acc = 0.4778
    hybrid_slow_acc = 0.7091
    
    if hybrid_summary.exists():
        with open(hybrid_summary) as f:
            data = json.load(f)
            hybrid_acc = data.get("accuracy", 0.605)
            hybrid_fast = data.get("fast_path_ratio", 0.45)
            hybrid_slow = data.get("slow_path_ratio", 0.55)
            hybrid_fast_acc = data.get("fast_path_accuracy", 0.4778)
            hybrid_slow_acc = data.get("slow_path_accuracy", 0.7091)
    
    models = [
        {
            "Model": "Random Baseline",
            "Assignment": "-",
            "Type": "Baseline",
            "Owner": "-",
            "Accuracy": 0.500,
            "Cost_Per_200": "$0.00",
            "Notes": "Theoretical lower bound"
        },
        {
            "Model": "TF-IDF + Cosine Similarity",
            "Assignment": "A1",
            "Type": "Lexical",
            "Owner": "Musaab",
            "Accuracy": 0.525,
            "Cost_Per_200": "$0.00",
            "Notes": "Simple baseline for error analysis"
        },
        {
            "Model": "SBERT Pretrained",
            "Assignment": "A2",
            "Type": "Embedding",
            "Owner": "Ahmed",
            "Accuracy": 0.550,
            "Cost_Per_200": "$0.00",
            "Notes": "No fine-tuning (all-MiniLM-L6-v2)"
        },
        {
            "Model": "TF-IDF + Logistic Regression",
            "Assignment": "A2",
            "Type": "ML Pipeline",
            "Owner": "Abdul Mueed",
            "Accuracy": 0.600,
            "Cost_Per_200": "$0.00",
            "Notes": "Trained on combined data"
        },
        {
            "Model": "Fine-tuned SBERT",
            "Assignment": "A2",
            "Type": "Embedding",
            "Owner": "Usman",
            "Accuracy": 0.585,
            "Cost_Per_200": "$0.00",
            "Notes": "Contrastive learning on synthetic data"
        },
        {
            "Model": "CoT LLM (GPT-4o-mini)",
            "Assignment": "A2",
            "Type": "LLM",
            "Owner": "Musaab",
            "Accuracy": 0.7181,
            "Cost_Per_200": "$0.20",
            "Notes": "Best accuracy, highest cost"
        },
        {
            "Model": "CoT LLM (Mistral Small)",
            "Assignment": "A2",
            "Type": "LLM",
            "Owner": "Musaab",
            "Accuracy": 0.690,
            "Cost_Per_200": "$0.15",
            "Notes": "Slightly lower accuracy, lower cost"
        },
        {
            "Model": "Multi-task SBERT",
            "Assignment": "A3",
            "Type": "Embedding",
            "Owner": "Ahmed/Usman",
            "Accuracy": 0.5025,
            "Cost_Per_200": "$0.00",
            "Notes": "FAILED - no generalization"
        },
        {
            "Model": "Hybrid (Proposed)",
            "Assignment": "A3",
            "Type": "Hybrid",
            "Owner": "Musaab",
            "Accuracy": hybrid_acc,
            "Cost_Per_200": "$0.11",
            "Notes": f"45% cost savings vs pure LLM"
        },
    ]
    
    # Save as CSV
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=models[0].keys())
        writer.writeheader()
        writer.writerows(models)
    
    print(f"✅ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*110)
    print("TABLE 1: MODEL COMPARISON ACROSS ALL ASSIGNMENTS")
    print("="*110)
    print(f"{'Model':<30} {'Assign':<6} {'Type':<10} {'Owner':<12} {'Accuracy':<10} {'Cost':<10} {'Notes':<30}")
    print("-"*110)
    for m in models:
        acc_str = f"{m['Accuracy']:.1%}" if isinstance(m['Accuracy'], float) else m['Accuracy']
        print(f"{m['Model']:<30} {m['Assignment']:<6} {m['Type']:<10} {m['Owner']:<12} {acc_str:<10} {m['Cost_Per_200']:<10} {m['Notes']:<30}")
    print("="*110)
    
    return models


# ============================================================================
# TABLE 2: HYBRID MODEL DETAILED ANALYSIS
# ============================================================================

def generate_hybrid_analysis():
    """Generate detailed hybrid model analysis table."""
    
    # Load actual results
    hybrid_summary = SCRIPT_DIR.parent / "output" / "musaab_hybrid" / "hybrid_summary_t0.1.json"
    
    if hybrid_summary.exists():
        with open(hybrid_summary) as f:
            data = json.load(f)
    else:
        # Default values from our run
        data = {
            "total": 200,
            "accuracy": 0.605,
            "fast_path": 90,
            "slow_path": 110,
            "fast_path_ratio": 0.45,
            "slow_path_ratio": 0.55,
            "fast_path_accuracy": 0.4778,
            "slow_path_accuracy": 0.7091,
            "llm_errors": 0,
            "confidence_stats": {
                "mean": 0.1138,
                "median": 0.0864,
                "stdev": 0.1008
            }
        }
    
    analysis = [
        {"Metric": "Total Samples", "Value": str(data.get('total', 200)), "Notes": "dev_track_a.jsonl"},
        {"Metric": "Overall Accuracy", "Value": f"{data.get('accuracy', 0.605):.2%}", "Notes": "Combined fast + slow paths"},
        {"Metric": "Fast Path Count", "Value": str(data.get('fast_path', 90)), "Notes": f"{data.get('fast_path_ratio', 0.45):.1%} of total"},
        {"Metric": "Fast Path Accuracy", "Value": f"{data.get('fast_path_accuracy', 0.4778):.2%}", "Notes": "Embedding model only"},
        {"Metric": "Slow Path Count", "Value": str(data.get('slow_path', 110)), "Notes": f"{data.get('slow_path_ratio', 0.55):.1%} of total"},
        {"Metric": "Slow Path Accuracy", "Value": f"{data.get('slow_path_accuracy', 0.7091):.2%}", "Notes": "LLM (Mistral Small)"},
        {"Metric": "LLM Errors", "Value": str(data.get('llm_errors', 0)), "Notes": "API failures"},
        {"Metric": "Confidence Threshold", "Value": "0.10", "Notes": "|sim_a - sim_b| threshold"},
        {"Metric": "Mean Confidence", "Value": f"{data.get('confidence_stats', {}).get('mean', 0.1138):.4f}", "Notes": "Average |sim_a - sim_b|"},
        {"Metric": "Cost vs Pure LLM", "Value": "45% savings", "Notes": "$0.11 vs $0.20 per 200 samples"},
    ]
    
    # Save as CSV
    csv_path = RESULTS_DIR / "hybrid_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=analysis[0].keys())
        writer.writeheader()
        writer.writerows(analysis)
    
    print(f"✅ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*80)
    print("TABLE 2: HYBRID MODEL DETAILED ANALYSIS")
    print("="*80)
    print(f"{'Metric':<25} {'Value':<20} {'Notes':<35}")
    print("-"*80)
    for a in analysis:
        print(f"{a['Metric']:<25} {a['Value']:<20} {a['Notes']:<35}")
    print("="*80)
    
    return analysis


# ============================================================================
# TABLE 3: AUTHOR CONTRIBUTIONS
# ============================================================================

def generate_contribution_table():
    """Generate author contribution table."""
    
    contributions = [
        {
            "Member": "Muhammad Musaab ul Haq",
            "Role": "Team Lead, System Integrator",
            "A1": "TF-IDF baseline, error analysis (25 cases)",
            "A2": "CoT LLM Reasoner (71.8%)",
            "A3": "Hybrid model design & implementation",
            "Code_Files": "model_1_llm_reasoner.py, musaab_hybrid_model.py, generate_plots.py",
            "Report": "Abstract, Intro, Hybrid Method, Discussion, Conclusion"
        },
        {
            "Member": "Abdul Mueed Habib Raja",
            "Role": "Data & Classic NLP",
            "A1": "Statistical analysis, distribution plots",
            "A2": "TF-IDF + LogReg pipeline (~60%)",
            "A3": "Data augmentation (500 samples)",
            "Code_Files": "abdul_mueed_stats.py, abdulmueed.ipynb",
            "Report": "Section 3.1 Data Augmentation"
        },
        {
            "Member": "Ahmed Hassan Raza",
            "Role": "Embedding Models",
            "A1": "N-gram analysis, vocabulary statistics",
            "A2": "SBERT baseline (~55%)",
            "A3": "Multi-task architecture design",
            "Code_Files": "ahmed_ngram_analysis.py, ahmed_multitask_model.py",
            "Report": "Section 3.2 Multi-task Learning"
        },
        {
            "Member": "Usman Amjad",
            "Role": "Experiments & Fine-tuning",
            "A1": "Human vs synthetic comparison",
            "A2": "Fine-tuned SBERT (58.5%)",
            "A3": "Multi-task training & evaluation",
            "Code_Files": "model_4_finetuned_sbert.py, usman_amjad_Assignment3.py, evaluate_model.py",
            "Report": "Section 4-5 Experiments & Results"
        },
    ]
    
    # Save as CSV
    csv_path = RESULTS_DIR / "author_contributions.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=contributions[0].keys())
        writer.writeheader()
        writer.writerows(contributions)
    
    print(f"✅ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*120)
    print("TABLE 3: AUTHOR CONTRIBUTIONS")
    print("="*120)
    print(f"{'Member':<25} {'Role':<25} {'A3 Contribution':<40} {'Report Section':<30}")
    print("-"*120)
    for c in contributions:
        print(f"{c['Member']:<25} {c['Role']:<25} {c['A3']:<40} {c['Report']:<30}")
    print("="*120)
    
    return contributions


# ============================================================================
# TABLE 4: MULTI-TASK MODEL FAILURE ANALYSIS
# ============================================================================

def generate_multitask_analysis():
    """Generate multi-task model failure analysis."""
    
    # Load evaluation results
    eval_file = SCRIPT_DIR / "experiments" / "EXP_001_Combined_Data" / "evaluation" / "dev_track_a" / "evaluation_results.json"
    
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
    else:
        eval_data = {
            "metrics": {"accuracy": 0.5025, "precision": 0.5013, "recall": 0.955, "f1_score": 0.6575},
            "similarity_stats": {
                "positive_pairs": {"mean": 0.3796, "std": 0.1416},
                "negative_pairs": {"mean": 0.3724, "std": 0.1396}
            },
            "best_threshold": 0.12
        }
    
    analysis = [
        {"Aspect": "Training Loss (Start)", "Value": "2.02", "Interpretation": "Model began learning"},
        {"Aspect": "Training Loss (End)", "Value": "0.82", "Interpretation": "Loss decreased 60%"},
        {"Aspect": "Evaluation Accuracy", "Value": "50.25%", "Interpretation": "Essentially random guessing"},
        {"Aspect": "Similar Pairs Mean Sim", "Value": f"{eval_data['similarity_stats']['positive_pairs']['mean']:.4f}", "Interpretation": "Should be HIGH"},
        {"Aspect": "Dissimilar Pairs Mean Sim", "Value": f"{eval_data['similarity_stats']['negative_pairs']['mean']:.4f}", "Interpretation": "Should be LOW"},
        {"Aspect": "Separation Gap", "Value": "0.0072", "Interpretation": "Almost no separation!"},
        {"Aspect": "Diagnosis", "Value": "Embedding Collapse", "Interpretation": "Model outputs similar embeddings for all inputs"},
        {"Aspect": "Root Cause", "Value": "Domain Mismatch", "Interpretation": "Synthetic training ≠ real movie plots"},
    ]
    
    # Save as CSV
    csv_path = RESULTS_DIR / "multitask_failure_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=analysis[0].keys())
        writer.writeheader()
        writer.writerows(analysis)
    
    print(f"✅ Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "="*90)
    print("TABLE 4: MULTI-TASK MODEL FAILURE ANALYSIS")
    print("="*90)
    print(f"{'Aspect':<30} {'Value':<20} {'Interpretation':<40}")
    print("-"*90)
    for a in analysis:
        print(f"{a['Aspect']:<30} {a['Value']:<20} {a['Interpretation']:<40}")
    print("="*90)
    
    return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("GENERATING ASSIGNMENT 3 RESULT TABLES")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"Output directory: {RESULTS_DIR}\n")
    
    generate_model_comparison()
    generate_hybrid_analysis()
    generate_contribution_table()
    generate_multitask_analysis()
    
    print("\n" + "="*70)
    print("ALL TABLES GENERATED!")
    print("="*70)
    print(f"\nFiles saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    for f in RESULTS_DIR.glob("*.csv"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()