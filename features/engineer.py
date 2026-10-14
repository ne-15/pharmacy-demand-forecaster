"""
Feature Engineering Module for Pharmacy Demand Forecasting.

Constructs comprehensive time-series regression features:
- Lag features (t-1, t-2, t-3, t-7, t-14, t-21, t-28)
- Rolling window aggregates with strict leakage prevention (shift(1))
- Exponential moving averages
- Cyclical calendar and seasonal representations (sine/cosine)
- Refill-cycle indicators and momentum ratios
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


LAG_DAYS = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]


class PharmacyFeatureEngineer:
    """
    Transforms raw time-series sales records into rich regression-ready feature sets.
    Ensures zero data leakage by shifting historical series before computing rolling metrics.
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        target_col: str = "units_sold"
    ):
        self.lags = lags if lags is not None else LAG_DAYS
        self.rolling_windows = rolling_windows if rolling_windows is not None else ROLLING_WINDOWS
        self.target_col = target_col
        self.feature_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame and records the resulting feature column names."""
        df_feat = self.transform(df)
        return df_feat

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Generates lag, rolling, seasonal, and momentum features per medicine.
        
        Args:
            df: Raw DataFrame containing at least 'date', 'medicine_id', 'units_sold'.
            drop_na: If True, drops initial warm-up rows containing NaN due to max lag/window.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["medicine_id", "date"]).reset_index(drop=True)

        feature_dfs = []

        for med_id, group in df.groupby("medicine_id", sort=False):
            group = group.copy().sort_values(by="date")
            target_series = group[self.target_col]

            # 1. Lag Features (e.g. sales 1 day ago, 7 days ago, etc.)
            for lag in self.lags:
                group[f"lag_{lag}"] = target_series.shift(lag)

            # 2. Rolling Window Aggregates (using shifted series to avoid leakage!)
            shifted_series = target_series.shift(1)
            for window in self.rolling_windows:
                group[f"roll_mean_{window}"] = shifted_series.rolling(window=window).mean()
                group[f"roll_std_{window}"] = shifted_series.rolling(window=window).std().fillna(0)
                group[f"roll_min_{window}"] = shifted_series.rolling(window=window).min()
                group[f"roll_max_{window}"] = shifted_series.rolling(window=window).max()

            # 3. Exponential Moving Averages
            group["ema_7"] = shifted_series.ewm(span=7, adjust=False).mean()
            group["ema_14"] = shifted_series.ewm(span=14, adjust=False).mean()

            # 4. Momentum and Acceleration indicators
            if "roll_mean_7" in group.columns and "roll_mean_28" in group.columns:
                group["demand_momentum_7_28"] = (group["roll_mean_7"] + 1e-4) / (group["roll_mean_28"] + 1e-4)
            if "lag_1" in group.columns and "lag_2" in group.columns:
                group["accel_day_over_day"] = group["lag_1"] - group["lag_2"]
            if "lag_1" in group.columns and "lag_7" in group.columns:
                group["diff_week_over_week"] = group["lag_1"] - group["lag_7"]

            feature_dfs.append(group)

        df_out = pd.concat(feature_dfs, ignore_index=True)

        # 5. Calendar and Seasonality features
        dates = pd.to_datetime(df_out["date"])
        df_out["day_of_week"] = dates.dt.dayofweek
        df_out["day_of_month"] = dates.dt.day
        df_out["month"] = dates.dt.month
        df_out["quarter"] = dates.dt.quarter
        df_out["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        
        # Monthly prescription refill indicators
        df_out["is_refill_window_start"] = (dates.dt.day <= 5).astype(int)
        df_out["is_refill_window_end"] = (dates.dt.day >= 27).astype(int)

        # Cyclical trigonometric transforms (preserves circular continuity)
        df_out["sin_day_of_week"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
        df_out["cos_day_of_week"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)
        df_out["sin_month"] = np.sin(2 * np.pi * (df_out["month"] - 1) / 12.0)
        df_out["cos_month"] = np.cos(2 * np.pi * (df_out["month"] - 1) / 12.0)
        
        doy = dates.dt.dayofyear
        df_out["sin_day_of_year"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
        df_out["cos_day_of_year"] = np.cos(2 * np.pi * (doy - 1) / 365.25)

        # Drop initial warm-up period if drop_na=True (max lag is usually 28)
        if drop_na:
            df_out = df_out.dropna().reset_index(drop=True)

        # Define candidate numerical feature list
        exclude_cols = ["date", "medicine_id", "medicine_name", "category", self.target_col, "revenue"]
        self.feature_columns = [c for c in df_out.columns if c not in exclude_cols]

        return df_out

    def get_feature_names(self) -> List[str]:
        """Returns the list of engineered predictor feature names."""
        return self.feature_columns
