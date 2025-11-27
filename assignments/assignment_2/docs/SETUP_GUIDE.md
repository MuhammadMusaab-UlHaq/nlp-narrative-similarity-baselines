# Setup Guide for Assignment 2 - Fine-Tuned Embedding Baseline
## Author: Usman Amjad

## Issue: MSYS2 Python Environment Incompatibility

The current virtual environment at `d:\AI_project\.venv` uses MSYS2/UCRT64 Python 3.12.7, which has limited package availability and SSL certificate issues that prevent installing PyTorch and numpy from PyPI.

## Solution: Use Standard Windows Python

### Option 1: Create New Virtual Environment with Windows Python (RECOMMENDED)

1. **Install Python 3.10 or 3.11 from python.org** (if not already installed)
   - Download from: https://www.python.org/downloads/windows/
   - Make sure to check "Add Python to PATH" during installation

2. **Create a new virtual environment:**
   ```powershell
   # Navigate to project root
   cd d:\AI_project\ai_sem_proj_semeval-2026-task-4-baselines
   
   # Create new venv with system Python
   python -m venv .venv_windows
   
   # Activate it
   .\.venv_windows\Scripts\Activate.ps1
   ```

3. **Install required packages:**
   ```powershell
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install PyTorch (CPU version for faster download)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # Install sentence-transformers and other dependencies
   pip install sentence-transformers pandas numpy scikit-learn matplotlib seaborn nltk
   ```

4. **Run the training script:**
   ```powershell
   python assignments\assignment_2\scripts\usman_finetune_embeddings.py
   ```

### Option 2: Use Conda (Alternative)

If you have Anaconda or Miniconda installed:

```powershell
# Create conda environment
conda create -n semeval python=3.10 -y
conda activate semeval

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other packages
pip install sentence-transformers pandas scikit-learn matplotlib seaborn nltk
```

### Option 3: Use Google Colab (If Local Installation Fails)

Upload the script and data to Google Colab which has all dependencies pre-installed.

## Expected Training Time

- **Data Loading:** ~5 seconds
- **Fine-Tuning (4 epochs):** ~15-30 minutes on CPU, ~5-10 minutes on GPU
- **Evaluation:** ~2-3 minutes
- **Total:** ~20-35 minutes

## Expected Results

- **Baseline (SBERT without fine-tuning):** ~65-70% accuracy
- **Fine-tuned Model:** ~72-78% accuracy
- **Improvement:** 5-10% expected

## Troubleshooting

### SSL Certificate Errors
If you encounter SSL errors:
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Out of Memory Errors
Reduce batch size in the script:
- Change `BATCH_SIZE = 16` to `BATCH_SIZE = 8` or `BATCH_SIZE = 4`

### CUDA Out of Memory
The script uses CPU by default. If you want to use GPU, ensure you have enough VRAM (at least 4GB recommended).
