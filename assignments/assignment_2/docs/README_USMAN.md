# Assignment 2: Fine-Tuned Embedding Baseline
**Author:** Usman Amjad  
**Task:** SemEval 2026 Task 4 - Semantic Proximity Detection

## Overview

This assignment implements a **Fine-Tuned Embedding Baseline** using contrastive learning. The goal is to improve upon the baseline SentenceTransformer model by fine-tuning it on synthetic data specifically designed for semantic similarity tasks.

## Approach

### 1. **Contrastive Learning**
We use triplet-based contrastive learning where each training instance consists of:
- **Anchor story**: Reference narrative
- **Similar story**: Semantically close narrative (positive example)
- **Dissimilar story**: Semantically distant narrative (negative example)

### 2. **Training Strategy**
- Convert triplets into pairs with similarity labels:
  - `(anchor, similar) → label = 1.0`
  - `(anchor, dissimilar) → label = 0.0`
- Use **CosineSimilarityLoss** to train the model
- Fine-tune a pre-trained `all-MiniLM-L6-v2` model

### 3. **Evaluation**
- Test on `dev_track_a.jsonl` (201 examples)
- For each instance, compute cosine similarity between anchor and two candidate texts
- Predict the text with higher similarity as being closer

## Data

### Training Data
- **File:** `data/processed/synthetic_data_for_contrastive_learning.jsonl`
- **Size:** ~2,400 triplets → 4,800 training pairs
- **Format:** Each line contains `anchor_story`, `similar_story`, `dissimilar_story`

### Evaluation Data
- **File:** `data/raw/dev_track_a.jsonl`
- **Size:** 201 examples
- **Format:** Each line contains `anchor_text`, `text_a`, `text_b`, `text_a_is_closer`

## Implementation

### Files Created

```
assignments/assignment_2/
├── scripts/
│   ├── usman_finetune_embeddings.py    # Main training script
│   └── usman_compare_models.py         # Comparison script
├── output/
│   └── usman_amjad/
│       ├── finetuned_model/            # Saved fine-tuned model
│       ├── results.json                # Training results
│       └── comparison_results.json     # Comparison with baseline
├── reports/
│   └── usman_amjad/                    # Analysis and documentation
├── setup.bat                           # Windows setup script
└── SETUP_GUIDE.md                      # Detailed setup instructions
```

### Key Components

#### `usman_finetune_embeddings.py`
Main training script with the following functions:

- `load_contrastive_data()`: Loads triplets from JSONL file
- `prepare_training_examples()`: Converts triplets to training pairs
- `finetune_model()`: Fine-tunes SentenceTransformer with contrastive loss
- `evaluate_model()`: Evaluates on dev set using cosine similarity
- `main()`: Complete training and evaluation pipeline

#### `usman_compare_models.py`
Comparison script that:
- Evaluates baseline (pre-trained) model
- Evaluates fine-tuned model
- Computes improvement metrics
- Saves comparison results

## Setup and Installation

### Prerequisites
- Python 3.10 or 3.11
- Windows OS with standard Python (not MSYS2)

### Quick Start

**Option 1: Using Setup Script (Recommended)**
```powershell
cd d:\AI_project\ai_sem_proj_semeval-2026-task-4-baselines\assignments\assignment_2
.\setup.bat
```

**Option 2: Manual Installation**
```powershell
# Create virtual environment
python -m venv .venv_windows
.\.venv_windows\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers pandas numpy scikit-learn matplotlib seaborn nltk
```

See `SETUP_GUIDE.md` for detailed instructions and troubleshooting.

## Usage

### 1. Train the Fine-Tuned Model
```powershell
python assignments\assignment_2\scripts\usman_finetune_embeddings.py
```

This will:
- Load ~2,400 training triplets
- Create 4,800 training pairs
- Fine-tune for 4 epochs (~20-30 minutes on CPU)
- Evaluate on dev set
- Save model to `output/usman_amjad/finetuned_model/`

### 2. Compare with Baseline
```powershell
python assignments\assignment_2\scripts\usman_compare_models.py
```

This will:
- Evaluate baseline model
- Evaluate fine-tuned model
- Display comparison metrics
- Save results to `comparison_results.json`

## Hyperparameters

Default configuration:
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Base model
BATCH_SIZE = 16                   # Training batch size
EPOCHS = 4                        # Number of epochs
WARMUP_STEPS = 100                # LR warmup steps
```

### Tuning Suggestions

**If training is slow:**
- Increase `BATCH_SIZE` to 32 (if memory allows)

**If getting poor results:**
- Increase `EPOCHS` to 6 or 8
- Try different base models: `all-mpnet-base-v2` (larger, more accurate)

**If out of memory:**
- Decrease `BATCH_SIZE` to 8 or 4

## Expected Results

Based on the synthetic data quality and size:

| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy | 65-70% | 72-78% | +5-10% |
| Training Time | - | 20-30 min | - |

## Research Background

### Key Concepts

1. **Sentence-BERT (SBERT)**
   - Modification of BERT for generating sentence embeddings
   - Uses siamese networks with pooling
   - Enables efficient semantic similarity computation

2. **Contrastive Learning**
   - Learns by contrasting similar and dissimilar examples
   - Pulls similar examples closer in embedding space
   - Pushes dissimilar examples farther apart

3. **Cosine Similarity Loss**
   - Measures angle between embeddings
   - Range: 0 (orthogonal) to 1 (identical)
   - Effective for semantic similarity tasks

### References

- Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Henderson et al. (2017). "Efficient Natural Language Response Suggestion for Smart Reply"
- SentenceTransformers Documentation: https://www.sbert.net/

## Troubleshooting

### Issue: Python Environment Not Compatible
**Solution:** Use standard Windows Python instead of MSYS2. See `SETUP_GUIDE.md`.

### Issue: SSL Certificate Errors
**Solution:** 
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size or use CPU only (default).

### Issue: Poor Performance
**Possible Causes:**
- Insufficient training epochs
- Data quality issues
- Inappropriate batch size
- Need different base model

## Next Steps

1. **Experiment with hyperparameters**
   - Try different batch sizes: 8, 16, 32
   - Vary epochs: 2, 4, 6, 8
   - Test different models: mpnet, roberta

2. **Analyze errors**
   - Which examples does the model get wrong?
   - Are there patterns in misclassifications?

3. **Improve training data**
   - Can synthetic data quality be improved?
   - Would more data help?

4. **Document findings**
   - Create visualizations of results
   - Write comprehensive analysis report

## Author Notes

This implementation follows best practices for fine-tuning SentenceTransformers:
- Proper data loading and validation
- Progress tracking and logging
- Model checkpointing
- Comprehensive evaluation
- Results persistence

For questions or issues, refer to the setup guide or consult the SentenceTransformers documentation.
