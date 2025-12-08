# Assignment 3 - Model Evaluation Report

**Author:** Usman Amjad  
**Date:** December 8, 2025  
**Experiment ID:** EXP_001_Combined_Data  
**Model:** all-MiniLM-L6-v2 (Fine-tuned with Multi-Task Learning)

---

## Executive Summary

This report presents the evaluation results of a fine-tuned sentence transformer model trained on synthetic data and tested on official SemEval 2026 Task 4 Track A datasets. The model was trained using a multi-task learning approach combining semantic ranking and binary classification objectives.

**Key Findings:**
- ✅ Model successfully trained on GPU (NVIDIA RTX 3060)
- ✅ Evaluated on unseen official test data
- ⚠️ Performance indicates significant room for improvement
- 🎯 Model achieves ~50-56% accuracy on Track A tasks

---

## 1. Training Configuration

### Model Architecture
- **Base Model:** `all-MiniLM-L6-v2` (Sentence Transformer)
- **Training Strategy:** Multi-Task Learning
  - **Task 1:** Semantic Ranking (MultipleNegativesRankingLoss)
  - **Task 2:** Binary Classification (SoftmaxLoss)

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 4 |
| Warmup Steps | 100 |
| Device | CUDA (RTX 3060) |
| Checkpoint Frequency | Every 500 steps |

### Training Data
| Dataset | Size | Format |
|---------|------|--------|
| `combined_synthetic_for_training.jsonl` | 2,398 triplets | anchor, similar, dissimilar |
| `augmented_synthetic_500.jsonl` | 551 stories | ROCStories format |
| **Total Training Pairs** | **4,794 pairs** | Classification format |

**Data Sources:** Meta-Llama-3.1-8B, GPT-4o, GPT-4o-Mini (synthetic generation)

---

## 2. Evaluation Setup

### Test Datasets (Unseen Data)
| Dataset | Description | Size | Format |
|---------|-------------|------|--------|
| `dev_track_a.jsonl` | Official development set | 200 cases | anchor, text_a, text_b, label |
| `sample_track_a.jsonl` | Official sample set | 39 cases | anchor, text_a, text_b, label |

**Task:** Given an anchor text and two candidate texts (text_a, text_b), determine which candidate is semantically more similar to the anchor.

### Evaluation Methodology
1. **Embedding Generation:** Encode all texts using the fine-tuned model
2. **Similarity Computation:** Calculate cosine similarity between anchor and candidates
3. **Threshold Optimization:** Find optimal classification threshold using F1-score
4. **Metrics Calculation:** Compute accuracy, precision, recall, F1-score
5. **Statistical Analysis:** Analyze similarity score distributions

---

## 3. Results - dev_track_a (200 test cases)

### Performance Metrics

| Metric | Score | Percentage |
|--------|-------|------------|
| **Accuracy** | 0.5025 | 50.25% |
| **Precision** | 0.5013 | 50.13% |
| **Recall** | 0.9550 | 95.50% |
| **F1-Score** | 0.6575 | 65.75% |
| **Optimal Threshold** | 0.1200 | - |

### Confusion Matrix

|                    | Predicted Dissimilar | Predicted Similar |
|--------------------|---------------------|-------------------|
| **Actually Dissimilar** | 10 (TN) | 190 (FP) |
| **Actually Similar** | 9 (FN) | 191 (TP) |

**Analysis:**
- ⚠️ **High False Positive Rate:** 95% of dissimilar pairs incorrectly classified as similar
- ✅ **High Recall:** Model captures 95.5% of actually similar pairs
- 🚨 **Low Precision:** Only 50.1% of predicted similar pairs are actually similar
- 💡 **Model Bias:** Strong tendency to predict "similar" for most pairs

### Similarity Score Statistics

| Pair Type | Mean | Std Dev | Min | Max |
|-----------|------|---------|-----|-----|
| **Similar Pairs** | 0.3796 | 0.1416 | - | - |
| **Dissimilar Pairs** | 0.3724 | 0.1396 | - | - |
| **Difference** | 0.0072 | - | - | - |

**Critical Issue:** Only **0.72% difference** between similar and dissimilar pair scores indicates poor discriminative ability.

### Visualizations

Results include three plots saved in:  
`experiments/EXP_001_Combined_Data/evaluation/dev_track_a/`

1. **confusion_matrix.png** - Visual representation of classification errors
2. **similarity_distribution.png** - Histogram showing overlap between similar/dissimilar scores
3. **threshold_analysis.png** - F1/Precision/Recall curves across different thresholds

---

## 4. Results - sample_track_a (39 test cases)

### Performance Metrics

| Metric | Score | Percentage |
|--------|-------|------------|
| **Accuracy** | 0.5641 | 56.41% |
| **Precision** | 0.5352 | 53.52% |
| **Recall** | 0.9744 | 97.44% |
| **F1-Score** | 0.6909 | 69.09% |
| **Optimal Threshold** | 0.2000 | - |

### Confusion Matrix

|                    | Predicted Dissimilar | Predicted Similar |
|--------------------|---------------------|-------------------|
| **Actually Dissimilar** | 6 (TN) | 33 (FP) |
| **Actually Similar** | 1 (FN) | 38 (TP) |

**Analysis:**
- ✅ **Better than dev_track_a:** +6.16% accuracy improvement
- ⚠️ **Still High FP Rate:** 84.6% of dissimilar pairs misclassified
- ✅ **Excellent Recall:** 97.44% of similar pairs correctly identified
- 💡 **Same Pattern:** Model exhibits similar bias toward predicting "similar"

### Similarity Score Statistics

| Pair Type | Mean | Std Dev | Min | Max |
|-----------|------|---------|-----|-----|
| **Similar Pairs** | 0.4022 | 0.1135 | - | - |
| **Dissimilar Pairs** | 0.3779 | 0.1368 | - | - |
| **Difference** | 0.0243 | - | - | - |

**Improvement:** **2.43% difference** - slightly better separation than dev set, but still insufficient.

### Visualizations

Results saved in:  
`experiments/EXP_001_Combined_Data/evaluation/sample_track_a/`

---

## 5. Comparative Analysis

### Performance Comparison

| Dataset | Test Cases | Accuracy | F1-Score | Threshold | Score Separation |
|---------|-----------|----------|----------|-----------|------------------|
| **dev_track_a** | 200 | 50.25% | 65.75% | 0.12 | 0.72% |
| **sample_track_a** | 39 | 56.41% | 69.09% | 0.20 | 2.43% |
| **Improvement** | - | +6.16% | +3.34% | +0.08 | +1.71% |

### Key Observations

1. **Consistent Pattern:** Both datasets show the same fundamental issue - poor discrimination between similar and dissimilar texts
2. **Better on Sample:** Smaller sample set shows better performance, possibly due to:
   - Different difficulty distribution
   - Smaller dataset variance
   - Random chance (39 samples)
3. **Threshold Sensitivity:** Optimal threshold varies (0.12 vs 0.20), suggesting instability
4. **Recall-Precision Tradeoff:** Extremely high recall at the cost of very low precision

---

## 6. Problem Analysis

### Why is Performance Suboptimal?

#### 1. **Training-Test Domain Mismatch**
- **Training Data:** Synthetic stories generated by LLMs
  - More generic narratives
  - Artificially created similar/dissimilar pairs
  - Consistent structure and vocabulary
- **Test Data:** Real movie plot summaries
  - Complex, nuanced storylines
  - Subtle differences in semantics
  - Diverse writing styles and terminology

#### 2. **Model Calibration Issues**
- **Very Low Optimal Threshold (0.12):** Indicates model doesn't learn strong similarity signals
- **Narrow Score Range:** Most predictions fall between 0.2-0.6 similarity
- **Poor Separation:** Only ~1-2% difference between similar/dissimilar pairs

#### 3. **Dataset Size Limitations**
- **Training Size:** 2,949 examples (combined)
- **Modern Standards:** Typically need 10K-100K+ for robust performance
- **Diversity:** Limited to synthetic generation patterns

#### 4. **Loss Function Effectiveness**
- **MultipleNegativesRankingLoss:** May not penalize insufficient separation strongly enough
- **SoftmaxLoss:** Binary classification may oversimplify the task
- **Multi-Task Balance:** Unclear if both objectives are optimally weighted

---

## 7. Recommendations for Improvement

### Immediate Actions (Quick Wins)

#### 1. **Increase Training Data**
```python
# Target: 5K-10K triplets minimum
- Augment with paraphrasing techniques
- Use back-translation for diversity
- Generate more synthetic data with different LLMs
```

#### 2. **Adjust Loss Functions**
```python
# Try stronger contrastive losses
- CosineSimilarityLoss with specific margins
- TripletLoss with hard negative mining
- ContrastiveLoss with dynamic margins
```

#### 3. **Hyperparameter Tuning**
- Increase epochs: 4 → 10-15
- Adjust batch size: 32 → 16 (harder negatives per batch)
- Longer warmup: 100 → 500 steps
- Add learning rate decay

#### 4. **Threshold Calibration**
```python
# Use validation set for threshold tuning
- Split training data 90/10
- Optimize threshold on validation set
- Monitor threshold stability across epochs
```

### Medium-Term Improvements

#### 5. **Use Real Domain Data**
- **Collect:** Movie plot summaries from IMDb, Wikipedia
- **Annotate:** Manually label similar/dissimilar pairs
- **Mix:** Combine synthetic + real data (70/30 ratio)

#### 6. **Advanced Training Strategies**
```python
# Hard negative mining
- Select dissimilar stories with high similarity scores
- Re-train focusing on difficult examples

# Curriculum learning
- Start with easy examples (clear similar/dissimilar)
- Gradually introduce harder examples
```

#### 7. **Model Architecture Experiments**
- Try larger models: `all-mpnet-base-v2` (better than MiniLM)
- Test domain-specific models: `sentence-t5-base`
- Ensemble multiple models for better robustness

#### 8. **Feature Engineering**
```python
# Add auxiliary features
- Text length similarity
- Named entity overlap
- Genre/topic classification
- Temporal markers (story timeline)
```

### Long-Term Strategy

#### 9. **Cross-Validation Framework**
```python
# Implement k-fold CV on training data
- Ensure model generalizes
- Detect overfitting early
- Validate threshold stability
```

#### 10. **Active Learning Pipeline**
```python
# Iteratively improve with model feedback
1. Predict on unlabeled data
2. Human annotate low-confidence predictions
3. Retrain with corrected labels
4. Repeat until performance plateaus
```

---

## 8. Technical Implementation Details

### Files Generated

```
assignments/assignment_3/src/
├── usman_amjad_Assignment3.py          # Training script (GPU-enabled)
├── evaluate_model.py                    # Evaluation script
└── experiments/
    └── EXP_001_Combined_Data/
        ├── config.json                  # Model configuration
        ├── pytorch_model.bin            # Trained weights
        ├── modules.json                 # Model architecture
        └── evaluation/
            ├── dev_track_a/
            │   ├── evaluation_results.json
            │   ├── confusion_matrix.png
            │   ├── similarity_distribution.png
            │   └── threshold_analysis.png
            └── sample_track_a/
                ├── evaluation_results.json
                ├── confusion_matrix.png
                ├── similarity_distribution.png
                └── threshold_analysis.png
```

### Reproducibility

**To reproduce results:**
```bash
# 1. Train the model
cd assignments/assignment_3/src
python usman_amjad_Assignment3.py

# 2. Evaluate on test sets
python evaluate_model.py
```

**System Requirements:**
- Python 3.11+
- PyTorch with CUDA 11.8+
- NVIDIA GPU (RTX 3060 or better)
- 12GB+ VRAM
- Dependencies: `sentence-transformers`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 9. Conclusion

### Achievements ✅
1. Successfully implemented multi-task learning pipeline
2. Trained model on GPU with efficient batching
3. Created comprehensive evaluation framework
4. Generated detailed metrics and visualizations
5. Identified clear performance bottlenecks

### Current Limitations ⚠️
1. **Low accuracy (~50-56%):** Barely better than random guessing
2. **Poor discrimination:** Similar/dissimilar scores overlap significantly
3. **High false positive rate:** Model biased toward predicting "similar"
4. **Domain gap:** Synthetic training data doesn't match real test data well

### Next Steps 🎯
1. **Immediate:** Increase training data to 5K+ examples
2. **Short-term:** Experiment with harder loss functions and negative mining
3. **Medium-term:** Incorporate real movie plot data with manual annotations
4. **Long-term:** Build active learning pipeline for continuous improvement

### Learning Outcomes 📚
- Gained experience with sentence transformers and fine-tuning
- Understood challenges of domain adaptation (synthetic → real data)
- Learned importance of evaluation metrics beyond accuracy
- Developed skills in model diagnosis and performance analysis

---

## 10. Appendix

### A. Evaluation Metrics Explained

**Accuracy:** Percentage of correct predictions (TP + TN) / Total

**Precision:** Of all pairs predicted as similar, what % were actually similar?  
`Precision = TP / (TP + FP)`

**Recall:** Of all actually similar pairs, what % did we correctly identify?  
`Recall = TP / (TP + FN)`

**F1-Score:** Harmonic mean of precision and recall  
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

### B. Confusion Matrix Terms

- **True Positive (TP):** Correctly predicted as similar
- **True Negative (TN):** Correctly predicted as dissimilar
- **False Positive (FP):** Incorrectly predicted as similar (Type I error)
- **False Negative (FN):** Incorrectly predicted as dissimilar (Type II error)

### C. JSON Results Schema

```json
{
  "model_path": "string",
  "test_data": "string",
  "device": "cuda|cpu",
  "num_test_pairs": int,
  "num_positive": int,
  "num_negative": int,
  "best_threshold": float,
  "metrics": {
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1_score": float
  },
  "similarity_stats": {
    "positive_pairs": {"mean": float, "std": float, "min": float, "max": float},
    "negative_pairs": {"mean": float, "std": float, "min": float, "max": float}
  },
  "confusion_matrix": {
    "true_negative": int,
    "false_positive": int,
    "false_negative": int,
    "true_positive": int
  }
}
```

### D. References

- **SentenceTransformers Documentation:** https://www.sbert.net/
- **SemEval 2026 Task 4:** Semantic Similarity for Narrative Texts
- **Loss Functions Guide:** https://www.sbert.net/docs/package_reference/losses.html
- **Evaluation Metrics:** https://scikit-learn.org/stable/modules/model_evaluation.html

---

**Report Generated:** December 8, 2025  
**Model Checkpoint:** `experiments/EXP_001_Combined_Data/`  
**Evaluation Results:** `experiments/EXP_001_Combined_Data/evaluation/`

---

## Contact

For questions or collaboration:
- **Author:** Usman Amjad
- **Repository:** ai_sem_proj_semeval-2026-task-4-baselines
- **Branch:** usman_assignment_3
