"""
Inventory Optimization and Safety Stock Engine for Pharmacy Operations.

Translates demand forecast predictions and error variances into actionable inventory recommendations:
- Service-level based Safety Stock calculation
- Dynamic Reorder Point (ROP) determination
- Stockout Risk Alerting and Expiry Risk metrics
- Optimal Replenishment Order Batch suggestions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.stats import norm


# Common pharmaceutical service levels and corresponding normal z-scores
SERVICE_LEVEL_Z = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
    0.999: 3.090,
}


class PharmacyInventoryOptimizer:
    """
    Computes inventory control parameters tailored for pharmacy medicines.
    Uses forecast errors and lead times to calculate resilient buffer stocks.
    """

    def __init__(self, default_service_level: float = 0.95):
        self.default_service_level = default_service_level

    def compute_inventory_policy(
        self,
        predicted_daily_demand: float,
        forecast_rmse: float,
        lead_time_days: int = 3,
        current_inventory: Optional[int] = None,
        service_level: Optional[float] = None,
        holding_cost_annual_pct: float = 0.20,
        unit_cost: float = 10.0,
        order_setup_cost: float = 25.0
    ) -> Dict[str, Any]:
        """
        Calculates safety stock, reorder point, stockout risk, and recommended batch order.
        
        Args:
            predicted_daily_demand: Average predicted units needed per day.
            forecast_rmse: Standard deviation of forecast residuals.
            lead_time_days: Replenishment supplier lead time in days.
            current_inventory: Current on-hand stock units (optional).
            service_level: Target fulfillment service level (default 0.95 = 95%).
            holding_cost_annual_pct: Annual inventory holding cost as % of unit cost.
            unit_cost: Cost per medicine unit.
            order_setup_cost: Fixed administrative/delivery cost per order.
            
        Returns:
            Dictionary with detailed inventory recommendations.
        """
        sl = service_level if service_level is not None else self.default_service_level
        # Get z-score
        z = norm.ppf(sl)

        # Lead time demand expectation
        lead_time_demand = predicted_daily_demand * lead_time_days

        # Standard deviation during lead time: sqrt(L) * sigma_forecast_error
        # (Assuming lead time has negligible variance or deterministic lead time)
        sigma_lead_time = float(np.sqrt(max(1, lead_time_days)) * forecast_rmse)

        # Safety Stock
        safety_stock = int(np.ceil(z * sigma_lead_time))

        # Reorder Point (ROP) = (Demand during Lead Time) + Safety Stock
        reorder_point = int(np.ceil(lead_time_demand + safety_stock))

        # Economic Order Quantity (EOQ) calculation: sqrt( (2 * AnnualDemand * SetupCost) / HoldingCostPerUnit )
        annual_demand = max(1.0, predicted_daily_demand * 365)
        holding_cost_per_unit = max(0.5, unit_cost * holding_cost_annual_pct)
        eoq = int(np.ceil(np.sqrt((2 * annual_demand * order_setup_cost) / holding_cost_per_unit)))

        # Evaluate current stock status if provided
        status = "HEALTHY"
        days_of_supply = None
        stockout_risk_pct = 0.0
        recommended_order_qty = 0

        if current_inventory is not None:
            days_of_supply = round(current_inventory / max(0.1, predicted_daily_demand), 1)

            # Stockout risk during lead time: probability demand exceeds current inventory
            if sigma_lead_time > 0:
                z_current = (current_inventory - lead_time_demand) / sigma_lead_time
                stockout_risk_pct = round(float((1.0 - norm.cdf(z_current)) * 100.0), 1)
            else:
                stockout_risk_pct = 100.0 if current_inventory < lead_time_demand else 0.0

            if current_inventory <= safety_stock:
                status = "CRITICAL_STOCKOUT_RISK"
                recommended_order_qty = max(eoq, reorder_point - current_inventory)
            elif current_inventory <= reorder_point:
                status = "REORDER_TRIGGERED"
                recommended_order_qty = max(eoq, reorder_point - current_inventory)
            elif current_inventory > reorder_point * 2.5:
                status = "OVERSTOCKED"
                recommended_order_qty = 0
            else:
                status = "OPTIMAL"
                recommended_order_qty = 0

        return {
            "predicted_daily_demand": round(predicted_daily_demand, 1),
            "lead_time_days": lead_time_days,
            "service_level_pct": round(sl * 100, 1),
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "eoq": eoq,
            "current_inventory": current_inventory,
            "days_of_supply": days_of_supply,
            "stockout_risk_pct": stockout_risk_pct,
            "status": status,
            "recommended_order_qty": recommended_order_qty,
        }
