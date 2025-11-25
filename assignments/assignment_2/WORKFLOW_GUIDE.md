# Complete Workflow Guide - Assignment 2
**Author:** Usman Amjad

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ASSIGNMENT 2 WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

STEP 1: ENVIRONMENT SETUP
┌──────────────────────────────────────────────────────────────┐
│ Run: .\setup.bat                                             │
│                                                              │
│ Creates: .venv_windows/                                      │
│ Installs: PyTorch, sentence-transformers, pandas, etc.      │
│ Time: ~5-10 minutes (depending on internet speed)           │
└──────────────────────────────────────────────────────────────┘
                            ↓
STEP 2: VERIFY INSTALLATION
┌──────────────────────────────────────────────────────────────┐
│ Run: python assignments\assignment_2\scripts\                │
│      test_installation.py                                    │
│                                                              │
│ Checks: ✓ Package imports                                   │
│         ✓ Data files exist                                  │
│         ✓ Model download works                              │
│         ℹ CUDA available? (optional)                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
STEP 3: TRAIN FINE-TUNED MODEL
┌──────────────────────────────────────────────────────────────┐
│ Run: python assignments\assignment_2\scripts\                │
│      usman_finetune_embeddings.py                           │
│                                                              │
│ Process:                                                     │
│   [1/4] Load 2,400 triplets → 4,800 pairs                  │
│   [2/4] Fine-tune for 4 epochs (~20-30 min)                │
│   [3/4] Evaluate on 201 dev examples                       │
│   [4/4] Save model + results                               │
│                                                              │
│ Output:                                                      │
│   - output/usman_amjad/finetuned_model/                    │
│   - output/usman_amjad/results.json                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
STEP 4: COMPARE WITH BASELINE
┌──────────────────────────────────────────────────────────────┐
│ Run: python assignments\assignment_2\scripts\                │
│      usman_compare_models.py                                │
│                                                              │
│ Evaluates:                                                   │
│   - Baseline model (pre-trained)                            │
│   - Fine-tuned model                                        │
│                                                              │
│ Shows:                                                       │
│   Baseline:    65-70% accuracy                              │
│   Fine-tuned:  72-78% accuracy                              │
│   Improvement: +5-10%                                       │
│                                                              │
│ Output:                                                      │
│   - output/usman_amjad/comparison_results.json             │
└──────────────────────────────────────────────────────────────┘
                            ↓
STEP 5: ANALYZE AND DOCUMENT
┌──────────────────────────────────────────────────────────────┐
│ Create report in: reports/usman_amjad/                      │
│                                                              │
│ Include:                                                     │
│   - Training curves / metrics                               │
│   - Error analysis                                          │
│   - Comparison with baseline                                │
│   - Insights and findings                                   │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
INPUT DATA                    TRAINING                   EVALUATION
─────────────────────────────────────────────────────────────────

synthetic_data_for_           ┌──────────────┐          dev_track_a.jsonl
contrastive_learning.jsonl    │              │          ┌──────────────┐
┌────────────────┐           │  Fine-tune   │          │ anchor_text  │
│ anchor_story   │──────────>│  SentenceBERT │          │ text_a       │
│ similar_story  │           │              │          │ text_b       │
│dissimilar_story│           │ 4 epochs     │          │ label        │
└────────────────┘           │ batch_sz: 16 │          └──────────────┘
                             │              │                │
2,400 triplets              └──────────────┘                │
     ↓                              ↓                       ↓
4,800 pairs                  ┌──────────────┐         ┌──────────────┐
(positive + negative)        │ Fine-tuned   │────────>│  Evaluation  │
                            │    Model     │         │              │
                            └──────────────┘         │ Cosine Sim   │
                                                      │ Prediction   │
                                                      └──────────────┘
                                                             ↓
                                                      ┌──────────────┐
                                                      │  Accuracy    │
                                                      │  72-78%      │
                                                      └──────────────┘
```

## File Organization

```
ai_sem_proj_semeval-2026-task-4-baselines/
│
├── data/
│   ├── processed/
│   │   └── synthetic_data_for_contrastive_learning.jsonl  ← Training data
│   └── raw/
│       └── dev_track_a.jsonl  ← Evaluation data
│
└── assignments/
    └── assignment_2/
        ├── scripts/
        │   ├── usman_finetune_embeddings.py  ← Main training script
        │   ├── usman_compare_models.py       ← Comparison script
        │   └── test_installation.py          ← Verification script
        │
        ├── output/
        │   └── usman_amjad/
        │       ├── finetuned_model/          ← Saved model (created after training)
        │       ├── results.json              ← Training results
        │       └── comparison_results.json   ← Comparison results
        │
        ├── reports/
        │   └── usman_amjad/                  ← Your analysis reports
        │
        ├── setup.bat                         ← Windows setup script
        ├── SETUP_GUIDE.md                    ← Detailed setup instructions
        ├── README_USMAN.md                   ← Complete documentation
        └── IMPLEMENTATION_SUMMARY.md         ← This summary
```

## Command Reference

### Essential Commands

```powershell
# 1. Setup (one-time)
cd d:\AI_project\ai_sem_proj_semeval-2026-task-4-baselines\assignments\assignment_2
.\setup.bat

# 2. Activate environment (each session)
.\.venv_windows\Scripts\Activate.ps1

# 3. Verify installation
python assignments\assignment_2\scripts\test_installation.py

# 4. Train model
python assignments\assignment_2\scripts\usman_finetune_embeddings.py

# 5. Compare results
python assignments\assignment_2\scripts\usman_compare_models.py
```

### Alternative: Manual Setup

```powershell
# Create virtual environment
python -m venv .venv_windows

# Activate
.\.venv_windows\Scripts\Activate.ps1

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other packages
pip install sentence-transformers pandas numpy scikit-learn matplotlib seaborn nltk
```

## Expected Timeline

| Task | Duration | Notes |
|------|----------|-------|
| Environment setup | 5-10 min | One-time |
| Installation verification | 1-2 min | Quick test |
| Model training | 20-30 min | On CPU |
| Evaluation | 2-3 min | On dev set |
| Comparison | 2-3 min | Both models |
| **Total** | **30-50 min** | First run |

## Key Concepts Explained

### 1. Contrastive Learning

**Goal:** Learn embeddings where similar items are close, dissimilar items are far

```
Before training:                After training:
 anchor                          anchor
   ○                               ●
similar dissimilar             similar   dissimilar
   ○        ○                     ●          ○
(random distances)           (optimized distances)
```

### 2. Triplet Structure

```
Triplet:
┌────────────┐
│   Anchor   │  Reference story
└────────────┘
      │
      ├─────> Similar Story    (positive, label = 1.0)
      └─────> Dissimilar Story (negative, label = 0.0)
```

### 3. Training Process

```
Input Pair                Loss Calculation         Weight Update
───────────────────────────────────────────────────────────────
(anchor, similar)         Loss = 1 - cos_sim      Reduce loss
     ↓                          ↓                       ↓
  Encode                   Backprop              Update weights
     ↓                          ↓                       ↓
[embedding_a]             Compute gradient       Model improves
[embedding_s]                   ↓                       ↓
     ↓                    Update optimizer      Better embeddings
  cos_sim                       ↓
     ↓                    Repeat for next batch
  Compare with label
```

### 4. Evaluation Process

```
Test Instance:
┌────────────┐
│   Anchor   │
└────────────┘
      │
      ├─────> Text A  →  Similarity A = 0.82
      └─────> Text B  →  Similarity B = 0.65
                ↓
         Prediction: A is closer (0.82 > 0.65)
                ↓
         Compare with ground truth label
                ↓
         Correct? → Update accuracy
```

## Troubleshooting Flowchart

```
Problem: Can't install packages
    │
    ├─> Using MSYS2 Python?
    │   └─> YES → Use standard Windows Python
    │       Install from python.org
    │
    ├─> SSL errors?
    │   └─> YES → Use --trusted-host flag
    │
    └─> Other error?
        └─> Check SETUP_GUIDE.md

Problem: Training is slow
    │
    ├─> Using CPU?
    │   └─> YES → Normal, 20-30 min expected
    │       Consider GPU if available
    │
    └─> Reduce batch size to 8

Problem: Low accuracy (<70%)
    │
    ├─> Increase epochs to 6-8
    ├─> Try different model (mpnet)
    └─> Check data quality

Problem: Out of memory
    │
    └─> Reduce BATCH_SIZE
        16 → 8 → 4
```

## Success Criteria

✅ **Environment:** All packages installed, test_installation.py passes  
✅ **Training:** Completes 4 epochs without errors  
✅ **Model:** Saved to output/usman_amjad/finetuned_model/  
✅ **Results:** JSON files created with metrics  
✅ **Accuracy:** 72-78% on dev set (improvement over baseline)  

## Quick Start for Beginners

1. **Open PowerShell** in the assignment_2 directory
2. **Run setup:** `.\setup.bat`
3. **Wait** for installation to complete
4. **Activate environment:** `.\.venv_windows\Scripts\Activate.ps1`
5. **Test:** `python scripts\test_installation.py`
6. **Train:** `python scripts\usman_finetune_embeddings.py`
7. **Compare:** `python scripts\usman_compare_models.py`
8. **Done!** Check `output/usman_amjad/` for results

## Next Steps After Training

1. **Analyze results:**
   - Check `results.json` for training metrics
   - Review `comparison_results.json` for improvements

2. **Error analysis:**
   - Which examples are misclassified?
   - Are there patterns?

3. **Experiment:**
   - Try different epochs: 2, 4, 6, 8
   - Try different batch sizes: 8, 16, 32
   - Try different models: mpnet-base, roberta-base

4. **Document findings:**
   - Create visualizations
   - Write analysis report
   - Compare with team members' approaches

## Support Resources

- **Setup issues:** See `SETUP_GUIDE.md`
- **Usage questions:** See `README_USMAN.md`
- **Implementation details:** See source code comments
- **SentenceTransformers docs:** https://www.sbert.net/
- **PyTorch docs:** https://pytorch.org/docs/

---

**Ready to start? Run:** `.\setup.bat`
