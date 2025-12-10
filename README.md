# SemEval-2026 Task 4: Narrative Story Similarity

**Course Project - Artificial Intelligence**  
**University of Engineering and Technology, Lahore**

## 📋 Overview

This repository contains our team's implementation for [SemEval-2026 Task 4: How Similar is Too Similar?](https://sites.google.com/view/semeval2026-task4/) - a shared task focused on detecting semantic proximity between narrative stories.

### Task Description

Given an **anchor story** and two candidate stories (**Text A** and **Text B**), determine which candidate is semantically more similar to the anchor. This requires understanding:
- Abstract thematic elements
- Narrative structure and plot arcs
- Story outcomes and conclusions
- Character dynamics and relationships

## 🏆 Key Results

| Model | Assignment | Accuracy | Cost/Sample |
|-------|------------|----------|-------------|
| Random Baseline | - | 50.00% | $0 |
| TF-IDF + Cosine | A1 | 52.50% | $0 |
| SBERT Pretrained | A2 | ~55% | $0 |
| Fine-tuned SBERT | A2 | 58.50% | $0 |
| **CoT LLM (GPT-4)** | A2 | **71.80%** | ~$0.02 |
| Multi-task Learning | A3 | 50.25% | $0 |
| **Hybrid System** | A3 | **69.50%** | ~$0.015 (27% savings) |

**Best Overall:** CoT LLM at 71.8% accuracy  
**Best Cost-Efficient:** Hybrid System at 69.5% with 27% cost savings

## 👥 Team Members

| Name | Role | Key Contributions |
|------|------|-------------------|
| **Muhammad Musaab ul Haq** | System Integrator | Hybrid model, error analysis, final reports |
| **Ahmed Hassan Raza** | ML Engineer | N-gram analysis, Multi-task learning model |
| **Abdul Mueed** | Data Scientist | Statistical analysis, Data augmentation |
| **Usman Amjad** | ML Engineer | Fine-tuned SBERT, Experimentation |

## 📁 Repository Structure

```
ai_sem_proj_semeval-2026-task-4-baselines/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                           # Original competition data
│   │   ├── dev_track_a.jsonl          # Development set (201 examples)
│   │   ├── dev_track_b.jsonl
│   │   └── sample_track_*.jsonl
│   └── processed/                     # Generated training data
│       ├── synthetic_data_for_contrastive_learning.jsonl
│       ├── augmented_synthetic_500.jsonl
│       └── combined_synthetic_for_training.jsonl
├── baseline_organizers/               # Official baselines (unchanged)
│   ├── track_a_baseline.py
│   └── track_b_baseline.py
└── assignments/
    ├── assignment_1/                  # Problem & Data Understanding
    ├── assignment_2/                  # Baseline Implementation
    └── assignment_3/                  # Proposed Solution
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadMusaab-UlHaq/ai_sem_proj_semeval-2026-task-4-baselines.git
cd ai_sem_proj_semeval-2026-task-4-baselines

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Models

#### Assignment 1: TF-IDF Baseline
```bash
jupyter notebook assignments/assignment_1/src/musaab_baseline.ipynb
```

#### Assignment 2: Fine-tuned SBERT
```bash
python assignments/assignment_2/src/model_4_finetuned_sbert.py
```

#### Assignment 3: Hybrid Model (Best)
```bash
python assignments/assignment_3/src/musaab_hybrid_model.py
```

## 📊 Assignment Details

### Assignment 1: Problem & Data Understanding

**Goal:** Analyze the dataset to understand its challenges and inform model design.

**Key Findings:**
1. **Lexical Traps:** Stories with high keyword overlap but different narratives
2. **Abstract Themes:** Similarity based on concepts, not surface words  
3. **Narrative Structure:** Story outcomes are crucial for determining similarity

**Files:**
- `assignments/assignment_1/src/` - Analysis scripts
- `assignments/assignment_1/reports/` - Detailed findings

---

### Assignment 2: Baseline Implementation

**Goal:** Implement and compare baseline approaches.

**Models Implemented:**
| Model | Owner | Accuracy |
|-------|-------|----------|
| TF-IDF + LogReg | Abdul Mueed | ~60% |
| SBERT Pretrained | Ahmed | ~55% |
| Fine-tuned SBERT | Usman | 58.5% |
| CoT LLM (GPT-4) | Musaab | **71.8%** |

**Files:**
- `assignments/assignment_2/src/` - Model implementations
- `assignments/assignment_2/notebooks/` - Jupyter experiments

---

### Assignment 3: Proposed Solution

**Goal:** Improve upon A2 baselines with novel approaches.

**Approaches:**
1. **Multi-task Learning** (Ahmed): Dual-head model with ranking + classification
2. **Data Augmentation** (Abdul Mueed): 500 additional synthetic samples
3. **Hybrid System** (Musaab): SBERT + LLM with confidence-based routing

**Best Configuration:**
- **Threshold:** 0.15
- **Accuracy:** 69.5%
- **Cost Savings:** 27% vs pure LLM
- **Strategy:** Use SBERT for high-confidence predictions, escalate to LLM otherwise

**Files:**
- `assignments/assignment_3/src/musaab_hybrid_model.py` - Hybrid implementation
- `assignments/assignment_3/plots/` - Visualizations (PDF)
- `assignments/assignment_3/results/` - Result tables (CSV)
- `assignments/assignment_3/reports/assignment_3_report.tex` - LaTeX report

## 📈 Visualizations

Generated plots are available in `assignments/assignment_3/plots/`:
- `accuracy_comparison.pdf` - All model accuracies
- `hybrid_architecture.pdf` - System architecture diagram
- `hybrid_path_distribution.pdf` - Routing decisions
- `confusion_matrix_multitask.pdf` - Multi-task model analysis
- `similarity_distribution_overlap.pdf` - SBERT score distributions
- `training_loss_curves.pdf` - Training dynamics

## 🔧 Configuration

### Environment Variables
```bash
# For LLM-based models (Assignment 2 & 3)
export POE_API_KEY="your_poe_api_key"
```

### Key Parameters
- **Hybrid threshold:** 0.15 (configurable in `musaab_hybrid_model.py`)
- **SBERT model:** `all-MiniLM-L6-v2`
- **LLM:** GPT-4 via Poe API

## 📝 Reports

Detailed reports for each assignment:
- Assignment 1: `assignments/assignment_1/reports/`
- Assignment 2: `assignments/assignment_2/reports/`
- Assignment 3: `assignments/assignment_3/reports/assignment_3_report.tex`

## 🙏 Acknowledgments

- SemEval-2026 Task 4 organizers for the dataset and baselines
- Sentence-Transformers library
- Poe API for LLM access

## 📄 License

This project is for educational purposes as part of university coursework.