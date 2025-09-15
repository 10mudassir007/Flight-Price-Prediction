# tests/test_model.py
import os
import pytest

def test_model_availability():
    """Ensure trained models are saved."""
    model_path = "models/random_forest.pkl"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

def test_encoders_availability():
    """Ensure encoders are saved."""
    encoders_dir = "models/encoders"
    assert os.path.exists(encoders_dir), "Encoders folder not found"
    assert len(os.listdir(encoders_dir)) > 0, "No encoders saved in models/encoders/"
