@echo off
REM Setup script for Assignment 2 - Usman Amjad
REM This script creates a proper Windows Python virtual environment and installs dependencies

echo ========================================
echo Assignment 2 Setup Script
echo Author: Usman Amjad
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv .venv_windows
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call .venv_windows\Scripts\activate.bat

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo Step 4: Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo WARNING: PyTorch installation failed, trying alternative method...
    pip install torch torchvision torchaudio
)

echo Step 5: Installing sentence-transformers and dependencies...
pip install sentence-transformers pandas numpy scikit-learn matplotlib seaborn nltk

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the training script:
echo   1. Activate environment: .venv_windows\Scripts\activate
echo   2. Run: python assignments\assignment_2\scripts\usman_finetune_embeddings.py
echo.
pause
