# tests/test_data.py
import pytest
import pandas as pd
from src.data import load_processed_dataset

def test_processed_data_columns():
    """Check if processed dataset has relevant columns."""
    df = load_processed_dataset()

    required_cols = [
        'airline','departure_time','stops','arrival_time','class','duration','price',
        'source_longitude','destination_longitude','source_latitude','destination_latitude','Distance'
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
