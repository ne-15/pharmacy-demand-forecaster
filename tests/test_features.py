"""
Unit tests for PharmacyFeatureEngineer.
"""

import unittest
import pandas as pd
import numpy as np
from features.engineer import PharmacyFeatureEngineer
from data.generator import generate_pharmacy_sales


class TestFeatureEngineer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = generate_pharmacy_sales(start_date="2025-01-01", end_date="2025-04-30", seed=42)

    def test_feature_creation_and_lags(self):
        engineer = PharmacyFeatureEngineer(lags=[1, 7], rolling_windows=[7, 14])
        df_feat = engineer.fit_transform(self.df)

        self.assertFalse(df_feat.empty)
        # Check lag columns
        self.assertIn("lag_1", df_feat.columns)
        self.assertIn("lag_7", df_feat.columns)
        # Check rolling statistics
        self.assertIn("roll_mean_7", df_feat.columns)
        self.assertIn("roll_std_7", df_feat.columns)
        self.assertIn("roll_mean_14", df_feat.columns)
        # Check cyclical calendar columns
        self.assertIn("sin_day_of_week", df_feat.columns)
        self.assertIn("cos_day_of_week", df_feat.columns)
        self.assertIn("sin_month", df_feat.columns)
        # Check no missing values when drop_na is True
        self.assertEqual(df_feat[engineer.get_feature_names()].isna().sum().sum(), 0)

    def test_leakage_prevention(self):
        # Verify that rolling mean at day t only uses values up to t-1
        single_med = self.df[self.df["medicine_id"] == "MED-001"].copy().sort_values("date").reset_index(drop=True)
        engineer = PharmacyFeatureEngineer(lags=[1], rolling_windows=[7])
        df_feat = engineer.transform(single_med, drop_na=False)

        # lag_1 at index 10 must equal units_sold at index 9
        idx = 10
        self.assertEqual(df_feat.loc[idx, "lag_1"], single_med.loc[idx - 1, "units_sold"])
        # roll_mean_7 at index 10 must equal mean of units_sold from idx-7 to idx-1
        expected_roll = single_med.loc[idx-7:idx-1, "units_sold"].mean()
        self.assertAlmostEqual(df_feat.loc[idx, "roll_mean_7"], expected_roll, places=4)


if __name__ == "__main__":
    unittest.main()
