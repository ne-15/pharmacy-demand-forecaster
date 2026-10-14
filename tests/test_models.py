"""
Unit tests for PharmacyDemandForecaster and model benchmarking.
"""

import unittest
import pandas as pd
from data.generator import generate_pharmacy_sales
from models.forecaster import PharmacyDemandForecaster, benchmark_all_models


class TestDemandForecaster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1 year of data for swift and accurate testing
        cls.df = generate_pharmacy_sales(start_date="2024-01-01", end_date="2024-12-31", seed=42)

    def test_xgboost_train_and_forecast(self):
        forecaster = PharmacyDemandForecaster(model_name="xgboost", test_days=30)
        res = forecaster.train(self.df, medicine_id="MED-001")

        self.assertIn("metrics", res)
        metrics = res["metrics"]
        self.assertGreater(metrics["r2"], 0.2)
        self.assertLess(metrics["wape"], 40.0)

        # Future forecasting
        future_df = forecaster.forecast_future(self.df, medicine_id="MED-001", horizon_days=7)
        self.assertEqual(len(future_df), 7)
        self.assertIn("predicted_demand", future_df.columns)
        self.assertIn("lower_ci", future_df.columns)
        self.assertIn("upper_ci", future_df.columns)
        self.assertTrue((future_df["predicted_demand"] >= 0).all())

    def test_model_benchmarking(self):
        # Benchmark for 1 medicine with a small test set
        bench_df = benchmark_all_models(self.df, medicine_id="MED-003", test_days=20)
        self.assertFalse(bench_df.empty)
        self.assertGreaterEqual(len(bench_df), 3)
        self.assertIn("RMSE (Units)", bench_df.columns)
        self.assertIn("R² Score", bench_df.columns)


if __name__ == "__main__":
    unittest.main()
