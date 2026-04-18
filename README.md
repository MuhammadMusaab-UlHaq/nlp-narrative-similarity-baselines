# SemEval-2026 Task 4: Narrative Story Similarity Baselines

**Course Project – Artificial Intelligence (NUST, Islamabad)**

## Project Overview

This repository contains baseline and course-assignment implementations for **SemEval-2026 Task 4: How Similar is Too Similar?**.  
The task compares an anchor story with two candidate stories and predicts which candidate is more semantically similar.

## Repository Structure

```text
ai_sem_proj_semeval-2026-task-4-baselines/
├── README.md
├── LICENSE
├── requirements.txt
├── experiment.log                    # historical run log kept for record
├── data/
│   ├── raw/
│   └── processed/
├── baseline_organizers/              # official organizer baselines
└── assignments/
    ├── assignment_1/
    ├── assignment_2/
    └── assignment_3/
```

## Setup

```bash
git clone https://github.com/MuhammadMusaab-UlHaq/ai_sem_proj_semeval-2026-task-4-baselines.git
cd ai_sem_proj_semeval-2026-task-4-baselines

python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run Existing Baselines

### Assignment 1 (TF-IDF baseline)
```bash
jupyter notebook assignments/assignment_1/src/musaab_baseline.ipynb
```

### Assignment 2 (Fine-tuned SBERT baseline)
```bash
python assignments/assignment_2/src/model_4_finetuned_sbert.py
```

### Assignment 3 (Hybrid system)
```bash
python assignments/assignment_3/src/musaab_hybrid_model.py
```

## Reproducibility Notes

- Dependency versions are pinned as lower bounds in `requirements.txt`.
- Raw and processed data are expected under `data/raw` and `data/processed`.
- LLM-based runs require `POE_API_KEY`.
- Reported results come from assignment experiments and associated project reports.

## Key Reported Results

| Model | Assignment | Accuracy | Cost/Sample |
|---|---|---:|---:|
| Random Baseline | - | 50.00% | $0 |
| TF-IDF + Cosine | A1 | 52.50% | $0 |
| SBERT Pretrained | A2 | ~55% | $0 |
| Fine-tuned SBERT | A2 | 58.50% | $0 |
| **CoT LLM (GPT-4)** | A2 | **71.80%** | ~$0.02 |
| Multi-task Learning | A3 | 50.25% | $0 |
| **Hybrid System** | A3 | **69.50%** | ~$0.015 (27% savings) |

## Acknowledgments

- SemEval-2026 Task 4 organizers for the dataset and baseline framing.
