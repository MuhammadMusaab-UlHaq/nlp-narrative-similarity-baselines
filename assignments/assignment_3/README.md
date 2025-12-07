# Assignment 3: Proposed Solution & Result Analysis

## Overview
This assignment implements an incremental improvement over our best baseline from Assignment 2, focusing on multi-task learning and data augmentation strategies for improved narrative similarity detection.

## Team Progress Status

| Team Member | Task | Status | Output |
|-------------|------|--------|--------|
| **Abdul Mueed** | Strategic Data Augmentation | ✅ Complete | 500 augmented samples |
| **Ahmed Hassan** | Multi-Task Model Implementation | ⚠️ Script ready, training pending | Multi-task learning script |
| **Usman Amjad** | Experimentation Lead | ❌ Blocked | Waiting for trained model |
| **Muhammad Musaab** | System Integrator & Hybrid Model | 🔄 In Progress | Hybrid model + final report |

## Directory Structure

```
assignment_3/
├── notebooks/
│   └── abdulmueed.ipynb          # Data augmentation notebook (generates 500 samples)
├── src/
│   └── ahmed_multitask_model.py  # Multi-task learning implementation
├── plots/                        # Visualizations (to be populated)
├── reports/                      # Analysis reports (to be populated)
└── README.md                     # This file
```

## Completed Work

### 1. Data Augmentation (Abdul Mueed) ✅
- **File**: `notebooks/abdulmueed.ipynb`
- **Output**: `data/processed/augmented_synthetic_500.jsonl`
- **Description**: Generated 500 high-quality synthetic samples targeting weaknesses from Assignment 1 error analysis
- **Status**: Complete and merged

### 2. Multi-Task Model (Ahmed Hassan) ⚠️
- **File**: `src/ahmed_multitask_model.py`
- **Description**: Implements multi-task learning with two objectives:
  - Head 1: Semantic Ranking (triplet loss)
  - Head 2: Binary Classification (cross-entropy loss)
- **Status**: Script complete, awaiting training execution

## Dependency Chain

```
[Original Synthetic Data] ─┐
                          ├─→ [Ahmed's Multi-Task Model] ─→ [Usman's Experiments] ─→ [Musaab's Hybrid System]
[Abdul's 500 Samples] ────┘
```

## How to Run

### 1. Train Multi-Task Model (Ahmed needs to run this)
```bash
cd assignments/assignment_3
python src/ahmed_multitask_model.py
```

**Expected outputs:**
- Trained model saved to `output/ahmed_multitask_model/`
- Training metrics and loss curves
- Evaluation results on dev set

### 2. Run Experiments (Usman - after Ahmed's model is ready)
```bash
# Will use Ahmed's trained model
# Train on combined dataset (original + augmented)
# Generate performance metrics and visualizations
```

### 3. Hybrid Model Integration (Musaab - after experiments)
```bash
# Combine embedding model with CoT LLM
# Implement confidence-based routing
# Final evaluation and report generation
```

## Data Files

- **Original synthetic data**: `data/processed/synthetic_data_for_contrastive_learning.jsonl` (2,400 triplets)
- **Augmented data**: `data/processed/augmented_synthetic_500.jsonl` (500 new samples)
- **Dev set**: `data/raw/dev_track_a.jsonl` (201 examples)

## Next Steps

### Immediate Actions Required:
1. **Ahmed**: Run the multi-task training script
   ```bash
   python assignments/assignment_3/src/ahmed_multitask_model.py
   ```

2. **Usman**: Once Ahmed's model is ready:
   - Combine datasets (original 2400 + augmented 500)
   - Train on combined data
   - Generate evaluation metrics
   - Create loss curves and performance visualizations

3. **Musaab**: Design and implement hybrid system:
   - Fast embedding model for initial prediction
   - Confidence scoring mechanism
   - CoT LLM fallback for low-confidence cases
   - System integration and final evaluation

### Expected Timeline:
- Ahmed's training: 1-2 hours (including debugging)
- Usman's experiments: 2-3 hours
- Musaab's integration: 3-4 hours
- Final report compilation: 2 hours

## Technical Notes

### Multi-Task Learning Architecture
- Base model: SentenceTransformer (likely all-MiniLM-L6-v2 or similar)
- Two training objectives running simultaneously:
  1. Triplet ranking loss (for semantic similarity)
  2. Binary classification loss (for A/B choice prediction)
- Shared encoder with task-specific heads

### Data Augmentation Strategy
- Abdul's 500 samples specifically target error cases from Assignment 1:
  - Stories with misleading lexical overlap
  - Abstract thematic similarities
  - Narrative structure variations

### Hybrid Model Design (Planned)
```
Input → Embedding Model → Confidence Score
           ↓
    High Confidence → Direct Prediction
           ↓
    Low Confidence → CoT LLM (GPT-4o-mini/Mistral)
           ↓
        Final Output
```

## Evaluation Metrics
- Primary: Accuracy on dev_track_a.jsonl
- Secondary: Confidence calibration, inference time
- Analysis: Error types, improvement over baselines

## Known Issues / Blockers
1. Ahmed's model training is pending - blocking downstream work
2. Usman hasn't started - waiting for trained model
3. GPU availability might affect training time

## Assignment Deliverables
As per project requirements:
1. **Report (3-4 pages)**: Including proposed method, experiments, results
2. **Python code**: All models and evaluation scripts
3. **Visualizations**: Loss curves, confusion matrices, performance comparisons (PDF format)

---
**Last Updated**: December 7, 2025
**Maintained by**: Muhammad Musaab ul Haq (Team Lead)