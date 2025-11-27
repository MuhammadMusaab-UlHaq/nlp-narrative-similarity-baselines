# ⚡ Quick Reference Card
**Assignment 2: Fine-Tuned Embedding Baseline**  
**Author:** Usman Amjad

## 🚀 Quick Start (3 Commands)

```powershell
# 1. Setup (one-time, 5-10 minutes)
cd d:\AI_project\ai_sem_proj_semeval-2026-task-4-baselines\assignments\assignment_2
.\setup.bat

# 2. Train (20-30 minutes)
.\.venv_windows\Scripts\Activate.ps1
python scripts\usman_finetune_embeddings.py

# 3. Compare (2-3 minutes)
python scripts\usman_compare_models.py
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `scripts/usman_finetune_embeddings.py` | Main training script |
| `scripts/usman_compare_models.py` | Compare baseline vs fine-tuned |
| `scripts/test_installation.py` | Verify dependencies |
| `setup.bat` | Automated installation |
| `SETUP_GUIDE.md` | Detailed setup help |
| `README_USMAN.md` | Complete documentation |
| `WORKFLOW_GUIDE.md` | Visual workflow guide |
| `IMPLEMENTATION_SUMMARY.md` | What was created |

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| Training Data | 2,400 triplets → 4,800 pairs |
| Dev Set | 201 examples |
| Baseline Accuracy | 65-70% |
| Fine-tuned Accuracy | 72-78% |
| Improvement | +5-10% |
| Training Time | 20-30 min (CPU) |

## 🔧 Hyperparameters

```python
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 4
WARMUP_STEPS = 100
```

## ⚠️ Important: Python Environment

**Issue:** Current `.venv` uses MSYS2 Python (incompatible)  
**Solution:** Use standard Windows Python via `setup.bat`

## 📚 Documentation Quick Links

- **Need help installing?** → `SETUP_GUIDE.md`
- **Want full details?** → `README_USMAN.md`
- **Visual workflow?** → `WORKFLOW_GUIDE.md`
- **What was built?** → `IMPLEMENTATION_SUMMARY.md`

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Can't install PyTorch | Use Windows Python, not MSYS2 |
| SSL errors | Add `--trusted-host pypi.org` |
| Out of memory | Reduce BATCH_SIZE to 8 |
| Slow training | Normal on CPU (~30 min) |

## ✅ Success Checklist

- [ ] Run `setup.bat` successfully
- [ ] `test_installation.py` passes all tests
- [ ] Training completes 4 epochs
- [ ] Model saved to `output/usman_amjad/finetuned_model/`
- [ ] Accuracy 72-78% achieved
- [ ] Comparison shows improvement

## 🎯 What You're Implementing

**Contrastive Learning** for semantic similarity:
- Load synthetic triplets (anchor, similar, dissimilar)
- Fine-tune SentenceBERT with contrastive loss
- Evaluate on semantic proximity task
- Compare with baseline to measure improvement

## 💡 Key Concepts

1. **Triplet:** (anchor, similar, dissimilar)
2. **Training Pairs:** (anchor, similar)=1.0, (anchor, dissimilar)=0.0
3. **Loss:** CosineSimilarityLoss
4. **Evaluation:** Cosine similarity comparison

## 📈 Output Files

After training, check:
```
output/usman_amjad/
├── finetuned_model/          # Fine-tuned model
├── results.json              # Training metrics
└── comparison_results.json   # Baseline vs fine-tuned
```

## 🚨 Current Status

✅ All implementation files created  
✅ Complete documentation written  
⏳ **NEXT STEP:** Run `setup.bat` to install dependencies  
⏳ **THEN:** Run training script

## 📞 Help

Stuck? Check documentation in this order:
1. This quick reference
2. `SETUP_GUIDE.md` for installation issues
3. `README_USMAN.md` for detailed explanations
4. `WORKFLOW_GUIDE.md` for visual walkthrough

---
**Ready?** → `.\setup.bat` 🚀
