import os
from src.data_pipeline import load_and_prep_cargo_data

def test_cargo_data_loading():
    csv_path = "data/flights.csv"
    if os.path.exists(csv_path):
        df = load_and_prep_cargo_data(csv_path)
        assert df is not None
        assert not df.empty
        assert "Origin" in df.columns