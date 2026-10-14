"""
Unit tests for data generation and data loader modules.
"""

import os
import unittest
import pandas as pd
from data.generator import generate_pharmacy_sales, MEDICINE_CATALOG
from data.loader import load_and_validate_data


class TestDataGenerator(unittest.TestCase):

    def test_generate_pharmacy_sales_structure(self):
        # Generate smaller date range for swift unit testing
        df = generate_pharmacy_sales(start_date="2025-01-01", end_date="2025-03-31", seed=42)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        expected_cols = ["date", "medicine_id", "units_sold", "unit_price", "revenue", "category"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Ensure units sold is non-negative
        self.assertTrue((df["units_sold"] >= 0).all())
        
        # Verify multiple medicines
        self.assertEqual(df["medicine_id"].nunique(), len(MEDICINE_CATALOG))

    def test_load_and_validate_data(self):
        df_raw = generate_pharmacy_sales(start_date="2025-01-01", end_date="2025-01-31", seed=123)
        temp_csv = "temp_test_sales.csv"
        df_raw.to_csv(temp_csv, index=False)

        try:
            df_clean, meta = load_and_validate_data(temp_csv)
            self.assertIsInstance(df_clean, pd.DataFrame)
            self.assertEqual(meta["num_medicines"], len(MEDICINE_CATALOG))
            self.assertIn("start_date", meta)
            self.assertIn("end_date", meta)
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)


if __name__ == "__main__":
    unittest.main()
