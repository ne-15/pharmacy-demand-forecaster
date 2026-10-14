"""
Evaluation and diagnostic metrics for Pharmacy Demand Forecasting.
Includes retail supply-chain standards: WAPE, MAPE, RMSE, MAE, R2, and Tracking Bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes key performance indicators for regression models on demand forecasting.
    
    Returns:
        Dictionary of RMSE, MAE, MAPE (%), WAPE (%), R2, and Bias (mean error).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip negative predictions to zero (sales cannot be negative)
    y_pred = np.maximum(0.0, y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # WAPE: sum(|y - y_hat|) / sum(y)
    total_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (total_actual + 1e-8) * 100.0)

    # MAPE: avoid division by zero
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    # R2
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 0 else 0.0

    # Bias: positive means over-forecasting, negative means under-forecasting
    bias = float(np.mean(y_pred - y_true))

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "wape": round(wape, 2),
        "r2": round(r2, 4),
        "bias": round(bias, 2),
    }


def format_metrics_summary(metrics: Dict[str, float]) -> str:
    """Formats metrics nicely for logging or CLI display."""
    return (
        f"RMSE: {metrics['rmse']:.2f} units | "
        f"MAE: {metrics['mae']:.2f} units | "
        f"WAPE: {metrics['wape']:.2f}% | "
        f"MAPE: {metrics['mape']:.2f}% | "
        f"R²: {metrics['r2']:.4f} | "
        f"Bias: {metrics['bias']:+.2f}"
    )
