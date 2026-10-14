"""
Data loading and preprocessing utilities for Pharmacy Demand Forecaster.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional


REQUIRED_COLUMNS = ["date", "medicine_id", "units_sold"]


def load_and_validate_data(
    filepath_or_buffer,
    impute_missing_dates: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads pharmacy sales data from CSV, validates schema, ensures complete date ranges,
    and returns a clean DataFrame alongside metadata.
    
    Args:
        filepath_or_buffer: Path to CSV or file-like buffer (e.g. from Streamlit).
        impute_missing_dates: If True, reindexes each medicine's timeline to fill missing days with 0.
        
    Returns:
        (df_clean, metadata_dict)
    """
    if isinstance(filepath_or_buffer, str) and not os.path.exists(filepath_or_buffer):
        raise FileNotFoundError(f"File not found: {filepath_or_buffer}")

    df = pd.read_csv(filepath_or_buffer)

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded dataset is missing required columns: {missing_cols}")

    # Standardize column types
    df["date"] = pd.to_datetime(df["date"])
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)
    df["medicine_id"] = df["medicine_id"].astype(str)

    # Optional defaults if missing
    if "medicine_name" not in df.columns:
        df["medicine_name"] = df["medicine_id"]
    if "category" not in df.columns:
        df["category"] = "General"
    if "unit_price" not in df.columns:
        df["unit_price"] = 10.0
    if "is_promotional" not in df.columns:
        df["is_promotional"] = 0
    if "stockout_flag" not in df.columns:
        df["stockout_flag"] = 0
    if "temp_celsius" not in df.columns:
        df["temp_celsius"] = 22.0
    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = 0.0
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = 3
    if "is_outbreak_period" not in df.columns:
        df["is_outbreak_period"] = 0

    df = df.sort_values(by=["medicine_id", "date"]).reset_index(drop=True)

    # Impute missing dates per medicine if requested
    if impute_missing_dates:
        records = []
        for med_id, med_group in df.groupby("medicine_id"):
            med_group = med_group.drop_duplicates(subset=["date"])
            min_date = med_group["date"].min()
            max_date = med_group["date"].max()
            full_idx = pd.date_range(start=min_date, end=max_date, freq="D")
            
            med_group = med_group.set_index("date").reindex(full_idx)
            med_group["medicine_id"] = med_id
            med_group["medicine_name"] = med_group["medicine_name"].ffill().bfill()
            med_group["category"] = med_group["category"].ffill().bfill()
            med_group["unit_price"] = med_group["unit_price"].ffill().bfill()
            med_group["lead_time_days"] = med_group["lead_time_days"].ffill().bfill()
            med_group["is_promotional"] = med_group["is_promotional"].fillna(0).astype(int)
            med_group["stockout_flag"] = med_group["stockout_flag"].fillna(0).astype(int)
            med_group["is_outbreak_period"] = med_group["is_outbreak_period"].fillna(0).astype(int)
            med_group["temp_celsius"] = med_group["temp_celsius"].interpolate(method="linear").bfill().ffill()
            med_group["rainfall_mm"] = med_group["rainfall_mm"].fillna(0.0)
            med_group["units_sold"] = med_group["units_sold"].fillna(0).astype(int)
            
            med_group["date"] = full_idx
            records.append(med_group.reset_index(drop=True))

        df = pd.concat(records, ignore_index=True)

    metadata = {
        "num_records": len(df),
        "num_medicines": df["medicine_id"].nunique(),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "categories": sorted(df["category"].unique().tolist()),
        "medicines": df[["medicine_id", "medicine_name", "category", "lead_time_days"]].drop_duplicates().to_dict(orient="records"),
    }

    return df, metadata
