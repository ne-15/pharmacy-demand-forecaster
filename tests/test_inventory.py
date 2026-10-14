"""
Unit tests for PharmacyInventoryOptimizer.
"""

import unittest
from models.inventory_optimizer import PharmacyInventoryOptimizer


class TestInventoryOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = PharmacyInventoryOptimizer(default_service_level=0.95)

    def test_compute_inventory_policy(self):
        policy = self.optimizer.compute_inventory_policy(
            predicted_daily_demand=50.0,
            forecast_rmse=6.0,
            lead_time_days=4,
            current_inventory=120,
            service_level=0.95
        )

        self.assertIn("safety_stock", policy)
        self.assertIn("reorder_point", policy)
        self.assertIn("status", policy)
        
        # Lead time demand = 50 * 4 = 200
        # Reorder point should exceed 200 due to safety stock
        self.assertGreater(policy["reorder_point"], 200)
        self.assertGreater(policy["safety_stock"], 0)
        # With current_inventory = 120 < ROP (around 220), reorder must be triggered
        self.assertIn(policy["status"], ["REORDER_TRIGGERED", "CRITICAL_STOCKOUT_RISK"])
        self.assertGreater(policy["recommended_order_qty"], 0)


if __name__ == "__main__":
    unittest.main()
