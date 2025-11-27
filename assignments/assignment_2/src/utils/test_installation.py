"""
Quick Test Script - Verify Installation
Author: Usman Amjad

This script performs a quick sanity check to ensure all dependencies
are installed correctly before running the full training.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "SentenceTransformers"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")
    
    ROOT = Path(__file__).resolve().parents[4]
    
    files = [
        (ROOT / "data" / "processed" / "synthetic_data_for_contrastive_learning.jsonl", "Training data"),
        (ROOT / "data" / "raw" / "dev_track_a.jsonl", "Dev set"),
    ]
    
    failed = []
    for filepath, name in files:
        if filepath.exists():
            print(f"  ✓ {name}: {filepath}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {filepath}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_model_download():
    """Test that we can download a small model."""
    print("\nTesting model download...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("  Downloading test model (this may take a minute)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"  ✓ Model loaded and working")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
        return True, []
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False, [str(e)]


def test_cuda():
    """Test CUDA availability (optional)."""
    print("\nTesting CUDA (GPU support)...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  ℹ CUDA not available (will use CPU)")
            print("  ℹ Training will be slower but still functional")
        return True, []
    except Exception as e:
        print(f"  ⚠ Could not check CUDA: {e}")
        return True, []  # Not critical


def main():
    print("="*60)
    print("Installation Verification Test")
    print("Author: Usman Amjad")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Run tests
    passed, failed = test_imports()
    if not passed:
        print(f"\n⚠ Failed imports: {', '.join(failed)}")
        all_passed = False
    
    passed, failed = test_data_files()
    if not passed:
        print(f"\n⚠ Missing files: {', '.join(failed)}")
        all_passed = False
    
    if all_passed:
        # Only test model download if basic tests pass
        passed, failed = test_model_download()
        if not passed:
            all_passed = False
    
    # CUDA test (informational only)
    test_cuda()
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYou're ready to run the training script:")
        print("  python assignments\\assignment_2\\scripts\\usman_finetune_embeddings.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease:")
        print("  1. Install missing packages")
        print("  2. Check data file locations")
        print("  3. See SETUP_GUIDE.md for help")
    print()


if __name__ == "__main__":
    main()
