"""
Core Regression Forecaster Engine for Medicine Demand Prediction.

Supports:
- XGBoost, Random Forest, Gradient Boosting, Ridge Linear Regression
- Chronological Time-Series Validation (Zero Lookahead Leakage)
- Feature Importance and Feature Contribution Extraction
- Multi-Step Recursive Future Horizon Forecasting with Prediction Intervals
- Model Benchmarking and Comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features.engineer import PharmacyFeatureEngineer
from models.evaluator import calculate_metrics


MODEL_REGISTRY = {
    "xgboost": lambda: XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    ),
    "ridge": lambda: Ridge(alpha=1.0, random_state=42),
}


class PharmacyDemandForecaster:
    """
    Time-series regression forecaster utilizing lag, rolling, and seasonal features.
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        engineer: Optional[PharmacyFeatureEngineer] = None,
        test_days: int = 60
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
        
        self.model_name = model_name
        self.engineer = engineer if engineer is not None else PharmacyFeatureEngineer()
        self.test_days = test_days
        
        self.model = MODEL_REGISTRY[model_name]()
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.is_trained: bool = False
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        medicine_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Filters data for a specific medicine (or all) and generates engineered lag features.
        """
        df_target = df.copy()
        if medicine_id is not None:
            df_target = df_target[df_target["medicine_id"] == medicine_id].copy()

        df_feat = self.engineer.fit_transform(df_target)
        self.feature_names = self.engineer.get_feature_names()
        self.feature_df = df_feat

        # Chronological Train-Test Split (Last `test_days` rows as Holdout Test Set)
        dates = pd.to_datetime(df_feat["date"])
        split_date = dates.max() - pd.Timedelta(days=self.test_days)

        train_data = df_feat[dates < split_date].copy()
        test_data = df_feat[dates >= split_date].copy()

        return train_data, test_data, self.feature_names

    def train(
        self,
        df: pd.DataFrame,
        medicine_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trains the forecaster and evaluates performance on held-out out-of-time test set.
        """
        train_data, test_data, feature_cols = self.prepare_data(df, medicine_id=medicine_id)
        self.train_df = train_data
        self.test_df = test_data

        if len(train_data) < 30:
            raise ValueError(f"Insufficient training observations: {len(train_data)}. Need at least 30.")

        X_train = train_data[feature_cols].values
        y_train = train_data["units_sold"].values

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on Test Set
        if len(test_data) > 0:
            X_test = test_data[feature_cols].values
            y_test = test_data["units_sold"].values
            y_pred_test = np.maximum(0, self.model.predict(X_test))
            self.metrics = calculate_metrics(y_test, y_pred_test)
            test_predictions = y_pred_test
        else:
            y_pred_train = np.maximum(0, self.model.predict(X_train))
            self.metrics = calculate_metrics(y_train, y_pred_train)
            test_predictions = np.array([])

        feature_importance = self.get_feature_importances()

        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "num_train_samples": len(train_data),
            "num_test_samples": len(test_data),
            "feature_importances": feature_importance,
            "test_actuals": test_data["units_sold"].values if len(test_data) > 0 else np.array([]),
            "test_predictions": test_predictions,
            "test_dates": test_data["date"].dt.strftime("%Y-%m-%d").values if len(test_data) > 0 else np.array([])
        }

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """Retrieves and ranks feature importances."""
        if not self.is_trained:
            return pd.DataFrame()

        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)

        if importances is None:
            return pd.DataFrame()

        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return imp_df.head(top_n)

    def forecast_future(
        self,
        df_history: pd.DataFrame,
        medicine_id: str,
        horizon_days: int = 14,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Recursively projects demand for the next `horizon_days` into the future.
        Updates lag and rolling features iteratively with each predicted step.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before forecasting.")

        df_med = df_history[df_history["medicine_id"] == medicine_id].copy()
        df_med["date"] = pd.to_datetime(df_med["date"])
        df_med = df_med.sort_values(by="date").reset_index(drop=True)

        last_date = df_med["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

        # Get metadata constants for this medicine
        med_name = df_med["medicine_name"].iloc[-1]
        med_cat = df_med["category"].iloc[-1]
        unit_price = df_med["unit_price"].iloc[-1]
        lead_time = df_med["lead_time_days"].iloc[-1]

        # Uncertainty standard error multiplier (e.g. 1.96 for 95% interval)
        z_score = 1.96 if confidence_level == 0.95 else 1.645
        rmse = self.metrics.get("rmse", 5.0)

        # Working buffer containing historical + iteratively appended forecast records
        working_history = df_med.copy()

        predictions = []

        for i, f_date in enumerate(future_dates):
            # Compute features for the current snapshot including the newly predicted rows
            eng = PharmacyFeatureEngineer(
                lags=self.engineer.lags,
                rolling_windows=self.engineer.rolling_windows
            )
            working_feat = eng.transform(working_history, drop_na=False)
            
            # Extract row corresponding to the day before, or synthesize features for today
            latest_row = working_feat.iloc[-1]
            
            # Construct features for f_date
            dow = f_date.dayofweek
            dom = f_date.day
            month = f_date.month
            doy = f_date.dayofyear

            feature_dict = {}
            for col in self.feature_names:
                if col.startswith("lag_"):
                    lag_num = int(col.split("_")[1])
                    if lag_num <= len(working_history):
                        feature_dict[col] = float(working_history["units_sold"].iloc[-lag_num])
                    else:
                        feature_dict[col] = float(working_history["units_sold"].mean())
                elif col.startswith("roll_mean_"):
                    w = int(col.split("_")[2])
                    feature_dict[col] = float(working_history["units_sold"].iloc[-w:].mean())
                elif col.startswith("roll_std_"):
                    w = int(col.split("_")[2])
                    val = working_history["units_sold"].iloc[-w:].std()
                    feature_dict[col] = float(val) if not np.isnan(val) else 0.0
                elif col.startswith("roll_min_"):
                    w = int(col.split("_")[2])
                    feature_dict[col] = float(working_history["units_sold"].iloc[-w:].min())
                elif col.startswith("roll_max_"):
                    w = int(col.split("_")[2])
                    feature_dict[col] = float(working_history["units_sold"].iloc[-w:].max())
                elif col == "ema_7":
                    feature_dict[col] = float(working_history["units_sold"].ewm(span=7).mean().iloc[-1])
                elif col == "ema_14":
                    feature_dict[col] = float(working_history["units_sold"].ewm(span=14).mean().iloc[-1])
                elif col == "demand_momentum_7_28":
                    m7 = float(working_history["units_sold"].iloc[-7:].mean())
                    m28 = float(working_history["units_sold"].iloc[-28:].mean())
                    feature_dict[col] = (m7 + 1e-4) / (m28 + 1e-4)
                elif col == "accel_day_over_day":
                    l1 = float(working_history["units_sold"].iloc[-1])
                    l2 = float(working_history["units_sold"].iloc[-2]) if len(working_history) >= 2 else l1
                    feature_dict[col] = l1 - l2
                elif col == "diff_week_over_week":
                    l1 = float(working_history["units_sold"].iloc[-1])
                    l7 = float(working_history["units_sold"].iloc[-7]) if len(working_history) >= 7 else l1
                    feature_dict[col] = l1 - l7
                elif col == "day_of_week":
                    feature_dict[col] = dow
                elif col == "day_of_month":
                    feature_dict[col] = dom
                elif col == "month":
                    feature_dict[col] = month
                elif col == "quarter":
                    feature_dict[col] = (month - 1) // 3 + 1
                elif col == "is_weekend":
                    feature_dict[col] = 1 if dow in [5, 6] else 0
                elif col == "is_refill_window_start":
                    feature_dict[col] = 1 if dom <= 5 else 0
                elif col == "is_refill_window_end":
                    feature_dict[col] = 1 if dom >= 27 else 0
                elif col == "sin_day_of_week":
                    feature_dict[col] = np.sin(2 * np.pi * dow / 7.0)
                elif col == "cos_day_of_week":
                    feature_dict[col] = np.cos(2 * np.pi * dow / 7.0)
                elif col == "sin_month":
                    feature_dict[col] = np.sin(2 * np.pi * (month - 1) / 12.0)
                elif col == "cos_month":
                    feature_dict[col] = np.cos(2 * np.pi * (month - 1) / 12.0)
                elif col == "sin_day_of_year":
                    feature_dict[col] = np.sin(2 * np.pi * (doy - 1) / 365.25)
                elif col == "cos_day_of_year":
                    feature_dict[col] = np.cos(2 * np.pi * (doy - 1) / 365.25)
                elif col == "is_promotional":
                    feature_dict[col] = 0
                elif col == "stockout_flag":
                    feature_dict[col] = 0
                elif col == "is_outbreak_period":
                    feature_dict[col] = 0
                elif col == "temp_celsius":
                    feature_dict[col] = float(working_history["temp_celsius"].iloc[-1])
                elif col == "rainfall_mm":
                    feature_dict[col] = float(working_history["rainfall_mm"].iloc[-1])
                elif col == "lead_time_days":
                    feature_dict[col] = lead_time
                elif col == "unit_price":
                    feature_dict[col] = unit_price
                else:
                    feature_dict[col] = 0.0

            # Predict next day demand
            feat_row = pd.DataFrame([feature_dict])[self.feature_names]
            pred_val = float(max(0.0, self.model.predict(feat_row.values)[0]))

            # Horizon error scaling: uncertainty increases slightly with forecast distance
            horizon_scale = np.sqrt(1.0 + 0.08 * i)
            margin = z_score * rmse * horizon_scale
            lower_bound = max(0.0, round(pred_val - margin, 1))
            upper_bound = round(pred_val + margin, 1)

            predictions.append({
                "date": f_date.strftime("%Y-%m-%d"),
                "day_name": f_date.strftime("%a"),
                "medicine_id": medicine_id,
                "medicine_name": med_name,
                "predicted_demand": round(pred_val, 1),
                "lower_ci": lower_bound,
                "upper_ci": upper_bound,
            })

            # Append new predicted row to history for recursive feeding
            new_record = {
                "date": f_date,
                "medicine_id": medicine_id,
                "medicine_name": med_name,
                "category": med_cat,
                "units_sold": int(round(pred_val)),
                "unit_price": unit_price,
                "revenue": round(pred_val * unit_price, 2),
                "is_promotional": 0,
                "stockout_flag": 0,
                "lead_time_days": lead_time,
                "temp_celsius": float(working_history["temp_celsius"].iloc[-1]),
                "rainfall_mm": 0.0,
                "is_outbreak_period": 0,
            }
            working_history = pd.concat([working_history, pd.DataFrame([new_record])], ignore_index=True)

        return pd.DataFrame(predictions)


def benchmark_all_models(
    df: pd.DataFrame,
    medicine_id: Optional[str] = None,
    test_days: int = 60
) -> pd.DataFrame:
    """
    Trains and compares all candidate models (XGBoost, Random Forest, Ridge, Gradient Boosting)
    on the exact same chronological test split.
    """
    results = []
    for model_name in MODEL_REGISTRY.keys():
        forecaster = PharmacyDemandForecaster(model_name=model_name, test_days=test_days)
        train_res = forecaster.train(df, medicine_id=medicine_id)
        m = train_res["metrics"]
        results.append({
            "Model": model_name.replace("_", " ").title(),
            "RMSE (Units)": m["rmse"],
            "MAE (Units)": m["mae"],
            "WAPE (%)": f"{m['wape']:.2f}%",
            "MAPE (%)": f"{m['mape']:.2f}%",
            "R² Score": m["r2"],
            "Bias": m["bias"],
        })

    return pd.DataFrame(results).sort_values(by="RMSE (Units)").reset_index(drop=True)
