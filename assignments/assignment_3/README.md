# Assignment 3: Proposed Solution & Result Analysis

> 📖 **See the main [README.md](../../README.md) for project overview and setup instructions.**

## Overview

This assignment implements improvements over Assignment 2 baselines, focusing on multi-task learning, data augmentation, and a hybrid embedding-LLM system.

## ✅ Final Results

| Approach | Owner | Accuracy | Status |
|----------|-------|----------|--------|
| Multi-task Learning | Ahmed | 50.25% | ❌ Failed |
| Data Augmentation | Abdul Mueed | +500 samples | ✅ Complete |
| **Hybrid System** | Musaab | **69.5%** | ✅ Best A3 |

### Hybrid System Performance
- **Best Threshold:** 0.15
- **Accuracy:** 69.5%
- **Cost Savings:** 27% vs pure LLM
- **Strategy:** SBERT for high-confidence → LLM for uncertain cases

## Team Contributions

| Team Member | Task | Status |
|-------------|------|--------|
| **Abdul Mueed** | Data Augmentation (500 samples) | ✅ Complete |
| **Ahmed Hassan** | Multi-Task Model | ✅ Complete |
| **Usman Amjad** | Experimentation & Analysis | ✅ Complete |
| **Muhammad Musaab** | Hybrid Model & Report | ✅ Complete |

## Directory Structure

```
assignment_3/
├── src/
│   ├── musaab_hybrid_model.py        # Hybrid SBERT+LLM system
│   ├── ahmed_multitask_model.py      # Multi-task learning
│   ├── usman_amjad_Assignment3.py    # Experimentation
│   ├── evaluate_model.py             # Evaluation utilities
│   ├── generate_tables.py            # Result table generation
│   ├── musaab_generate_plots.py      # Visualization generation
│   └── experiments/                   # Trained models
├── notebooks/
│   └── abdulmueed.ipynb              # Data augmentation
├── output/
│   └── musaab_hybrid/                # Hybrid model outputs
│       ├── threshold_analysis.json   # Best threshold analysis
│       └── hybrid_summary_t*.json    # Results per threshold
├── plots/                            # PDF visualizations
│   ├── accuracy_comparison.pdf
│   ├── hybrid_architecture.pdf
│   ├── hybrid_path_distribution.pdf
│   └── ...
├── results/                          # CSV result tables
│   ├── model_comparison.csv
│   ├── hybrid_analysis.csv
│   └── author_contributions.csv
└── reports/
    └── assignment_3_report.tex       # Final LaTeX report
```

## Quick Start

### Run Hybrid Model
```bash
python assignments/assignment_3/src/musaab_hybrid_model.py
```

### Generate Plots
```bash
python assignments/assignment_3/src/musaab_generate_plots.py
```

### Generate Result Tables
```bash
python assignments/assignment_3/src/generate_tables.py
```

## Key Findings

1. **Multi-task Learning Failed:** 50.25% accuracy (near random) - the dual-head approach didn't transfer well to this task
2. **Hybrid System Effective:** By routing confident SBERT predictions directly and using LLM only for uncertain cases, we achieve 69.5% accuracy with 27% cost savings
3. **Best Threshold:** 0.15 - balances accuracy vs cost optimally

## Technical Details

### Multi-Task Learning Architecture
- Base model: SentenceTransformer (`all-MiniLM-L6-v2`)
- Two training objectives:
  1. Triplet ranking loss (for semantic similarity)
  2. Binary classification loss (for A/B choice prediction)
- **Result:** 50.25% accuracy (failed - task mismatch)

### Hybrid Model Architecture
```
Input → SBERT Embedding → Similarity Scores → Confidence Check
                                                    ↓
                              High Confidence (>0.15) → Direct SBERT Prediction
                                                    ↓
                              Low Confidence (≤0.15) → CoT LLM (GPT-4)
                                                    ↓
                                              Final Output
```

### Data Augmentation
- 500 synthetic samples targeting error cases from Assignment 1
- Focus on: lexical traps, abstract themes, narrative structure

## Deliverables

- ✅ **Code:** `src/musaab_hybrid_model.py`, `src/ahmed_multitask_model.py`
- ✅ **Report:** `reports/assignment_3_report.tex` (LaTeX, 4 pages)
- ✅ **Visualizations:** 6 PDF plots in `plots/`
- ✅ **Results:** CSV tables in `results/`

---
**Last Updated**: December 11, 2025  
**Status**: ✅ Complete