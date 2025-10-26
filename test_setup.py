#!/usr/bin/env python3
"""Test script to validate the setup."""

import sys
import torch
import transformers
from datasets import Dataset

def test_basic_imports():
    """Test that all required packages can be imported."""
    try:
        import torch
        import transformers
        import datasets
        import trl
        import numpy as np
        import pandas as pd
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_torch_setup():
    """Test PyTorch setup."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(2, 3).to(device)
        y = torch.mm(x, x.T)
        print(f"✓ PyTorch working on device: {device}")
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test transformers library."""
    try:
        from transformers import AutoTokenizer
        # Use a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokens = tokenizer("Hello world!")
        print("✓ Transformers library working")
        return True
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RL-LLM setup...")
    tests = [test_basic_imports, test_torch_setup, test_transformers]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
