"""Generate comparison tables for Assignment 3 report."""

import json
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR
RESULTS_DIR.mkdir(exist_ok=True)

def generate_comparison_table():
    """Generate CSV comparison of all models."""
    
    results = [
        {
            "Model": "Random Baseline",
            "Assignment": "-",
            "Owner": "-",
            "Accuracy": "50.00%",
            "Cost/Sample": "$0",
            "Notes": "Theoretical baseline"
        },
        {
            "Model": "TF-IDF + Cosine",
            "Assignment": "A1",
            "Owner": "Musaab",
            "Accuracy": "52.50%",
            "Cost/Sample": "$0",
            "Notes": "Simple lexical baseline"
        },
        {
            "Model": "SBERT Pretrained",
            "Assignment": "A2",
            "Owner": "Ahmed",
            "Accuracy": "~55%",
            "Cost/Sample": "$0",
            "Notes": "No fine-tuning"
        },
        {
            "Model": "TF-IDF + LogReg",
            "Assignment": "A2",
            "Owner": "Abdul Mueed",
            "Accuracy": "~60%",
            "Cost/Sample": "$0",
            "Notes": "Trained classifier"
        },
        {
            "Model": "Fine-tuned SBERT",
            "Assignment": "A2",
            "Owner": "Usman",
            "Accuracy": "58.50%",
            "Cost/Sample": "$0",
            "Notes": "Contrastive learning"
        },
        {
            "Model": "CoT LLM (GPT-4o-mini)",
            "Assignment": "A2",
            "Owner": "Musaab",
            "Accuracy": "71.81%",
            "Cost/Sample": "~$0.001",
            "Notes": "Best accuracy, highest cost"
        },
        {
            "Model": "Multi-task Learning",
            "Assignment": "A3",
            "Owner": "Ahmed/Usman",
            "Accuracy": "50.25%",
            "Cost/Sample": "$0",
            "Notes": "FAILED - no generalization"
        },
        {
            "Model": "Hybrid (Proposed)",
            "Assignment": "A3",
            "Owner": "Musaab",
            "Accuracy": "~70%",
            "Cost/Sample": "~$0.0007",
            "Notes": "Near-LLM accuracy, 30% cost reduction"
        },
    ]
    
    # Save as CSV
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved: {csv_path}")
    
    # Also print as table
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    print(f"{'Model':<25} {'Assign':<8} {'Owner':<15} {'Accuracy':<12} {'Cost':<12} {'Notes':<30}")
    print("-"*100)
    for r in results:
        print(f"{r['Model']:<25} {r['Assignment']:<8} {r['Owner']:<15} "
              f"{r['Accuracy']:<12} {r['Cost/Sample']:<12} {r['Notes']:<30}")
    print("="*100)
    
    return results


def generate_contribution_table():
    """Generate author contribution table."""
    
    contributions = [
        {
            "Member": "Muhammad Musaab ul Haq",
            "A1_Contribution": "TF-IDF baseline, error analysis (25 cases)",
            "A2_Contribution": "CoT LLM Reasoner (71.8%)",
            "A3_Contribution": "Hybrid model design & implementation, report writing",
            "Code": "model_1_llm_reasoner.py, musaab_hybrid_model.py",
            "Report_Sections": "Abstract, Introduction, Hybrid Method, Discussion, Conclusion"
        },
        {
            "Member": "Abdul Mueed Habib Raja",
            "A1_Contribution": "Statistical analysis, distribution plots",
            "A2_Contribution": "TF-IDF + LogReg pipeline",
            "A3_Contribution": "Data augmentation (500 samples)",
            "Code": "abdulmueed.ipynb",
            "Report_Sections": "Section 3.1 Data Augmentation"
        },
        {
            "Member": "Ahmed Hassan Raza",
            "A1_Contribution": "N-gram analysis, vocabulary stats",
            "A2_Contribution": "SBERT baseline",
            "A3_Contribution": "Multi-task architecture",
            "Code": "ahmed_multitask_model.py",
            "Report_Sections": "Section 3.2 Multi-task Learning"
        },
        {
            "Member": "Usman Amjad",
            "A1_Contribution": "Human vs synthetic comparison",
            "A2_Contribution": "Fine-tuned SBERT (58.5%)",
            "A3_Contribution": "Training & evaluation, visualizations",
            "Code": "usman_amjad_Assignment3.py, evaluate_model.py",
            "Report_Sections": "Section 4-5 Experiments & Results"
        },
    ]
    
    csv_path = RESULTS_DIR / "author_contributions.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=contributions[0].keys())
        writer.writeheader()
        writer.writerows(contributions)
    
    print(f"\nSaved: {csv_path}")
    return contributions


if __name__ == "__main__":
    generate_comparison_table()
    generate_contribution_table()