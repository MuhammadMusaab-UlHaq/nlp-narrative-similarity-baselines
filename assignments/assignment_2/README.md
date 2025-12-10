# Assignment 2: Baseline Implementation

> 📖 **See the main [README.md](../../README.md) for project overview and setup instructions.**

## Overview

Implementation and comparison of baseline approaches for narrative story similarity detection.

## Results

| Model | Owner | Accuracy |
|-------|-------|----------|
| TF-IDF + LogReg | Abdul Mueed | ~60% |
| SBERT Pretrained | Ahmed | ~55% |
| Fine-tuned SBERT | Usman | 58.5% |
| **CoT LLM (GPT-4)** | Musaab | **71.8%** |

## Directory Structure

```
assignment_2/
├── src/
│   ├── model_1_llm_reasoner.py      # LLM-based reasoning
│   ├── model_3_official_sbert.py    # SBERT baseline
│   ├── model_4_finetuned_sbert.py   # Fine-tuned SBERT
│   └── utils/                        # Utility functions
├── notebooks/
│   ├── mueed_assignement2.ipynb     # Abdul Mueed's experiments
│   └── Usman_Assignment2_Colab.ipynb # Usman's Colab notebook
├── output/                           # Model outputs
└── reports/                          # Analysis reports
```

## Quick Start

```bash
# Run fine-tuned SBERT
python assignments/assignment_2/src/model_4_finetuned_sbert.py

# Run LLM reasoner (requires POE_API_KEY)
python assignments/assignment_2/src/model_1_llm_reasoner.py
```

## Key Findings

- **Best Model:** Chain-of-Thought LLM (GPT-4) achieved 71.8% accuracy
- **Cost Trade-off:** LLM is accurate but expensive (~$0.02/sample)
- **SBERT Limitation:** Fine-tuning improved accuracy but still below LLM
