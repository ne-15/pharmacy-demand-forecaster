"""
Command Line Interface (CLI) for Pharmacy Demand Forecaster.

Enables automated pipelines:
- Data generation & validation
- Model training & evaluation (XGBoost, Random Forest, Ridge, Gradient Boosting)
- Comprehensive benchmarking across models
- Future demand forecasting
- Inventory safety stock and reorder point recommendation
"""

import argparse
import sys
import os
import pandas as pd

from data.generator import save_dataset_if_not_exists, MEDICINE_CATALOG
from data.loader import load_and_validate_data
from models.forecaster import PharmacyDemandForecaster, benchmark_all_models
from models.evaluator import format_metrics_summary
from models.inventory_optimizer import PharmacyInventoryOptimizer


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Pharmacy Demand Forecaster - Time Series Regression with Lag Features"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw_pharmacy_sales.csv",
        help="Path to pharmacy sales CSV file."
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic multi-year pharmacy sales dataset."
    )
    parser.add_argument(
        "--medicine",
        type=str,
        default="MED-001",
        help="Medicine ID to model/forecast (e.g. MED-001, MED-003)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "gradient_boosting", "ridge"],
        help="Regression model architecture."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train forecaster and evaluate on held-out test split."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark all candidate regression models on the same test split."
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Generate recursive demand forecast for upcoming days."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Number of future days to forecast (default: 14)."
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Compute inventory reorder policy and safety stock recommendation."
    )
    parser.add_argument(
        "--current-stock",
        type=int,
        default=100,
        help="Current stock level in units for inventory calculation."
    )
    parser.add_argument(
        "--service-level",
        type=float,
        default=0.95,
        help="Desired fulfillment service level (default: 0.95 for 95 percent)."
    )

    args = parser.parse_args()

    # If no action is specified, show help
    if not any([args.generate_data, args.train, args.benchmark, args.forecast, args.inventory]):
        parser.print_help()
        print("\nTip: Run with --generate-data or --benchmark --medicine MED-001 to get started!")
        return

    # 1. Dataset Generation
    if args.generate_data or not os.path.exists(args.data):
        print_header("Dataset Preparation")
        data_path = save_dataset_if_not_exists(args.data, force_regenerate=args.generate_data)
    else:
        data_path = args.data

    # Load dataset
    df, meta = load_and_validate_data(data_path)
    print(f"\n[Data Loaded] Records: {meta['num_records']} | Medicines: {meta['num_medicines']} | Time Range: {meta['start_date']} to {meta['end_date']}")

    # 2. Benchmarking
    if args.benchmark:
        print_header(f"Model Benchmark Comparison (Medicine: {args.medicine})")
        bench_df = benchmark_all_models(df, medicine_id=args.medicine, test_days=60)
        print(bench_df.to_string(index=False))

    # 3. Training & Evaluation
    forecaster = PharmacyDemandForecaster(model_name=args.model, test_days=60)
    if args.train or args.forecast or args.inventory:
        print_header(f"Training Forecaster [{args.model.upper()}] (Medicine: {args.medicine})")
        res = forecaster.train(df, medicine_id=args.medicine)
        print(f"Training Observations: {res['num_train_samples']} | Test Observations: {res['num_test_samples']}")
        print(f"Evaluation Metrics: {format_metrics_summary(res['metrics'])}")
        
        print("\nTop 7 Engineered Feature Importances:")
        top_imp = res["feature_importances"].head(7)
        for _, r in top_imp.iterrows():
            print(f"  - {r['feature']:<25} : {r['importance']:.4f}")

    # 4. Multi-step Future Forecasting
    if args.forecast:
        print_header(f"Recursive Demand Forecast - Next {args.horizon} Days")
        future_df = forecaster.forecast_future(
            df,
            medicine_id=args.medicine,
            horizon_days=args.horizon
        )
        print(future_df[["date", "day_name", "predicted_demand", "lower_ci", "upper_ci"]].to_string(index=False))
        total_projected = future_df["predicted_demand"].sum()
        print(f"\nTotal Projected Units for Next {args.horizon} Days: {total_projected:.0f} units")

    # 5. Inventory Optimization
    if args.inventory:
        print_header("Pharmacy Inventory & Safety Stock Optimization")
        # Predict daily demand over the next 14 days
        future_df = forecaster.forecast_future(df, medicine_id=args.medicine, horizon_days=14)
        avg_daily_demand = future_df["predicted_demand"].mean()
        rmse = forecaster.metrics.get("rmse", 5.0)

        med_info = df[df["medicine_id"] == args.medicine].iloc[-1]
        lead_time = int(med_info.get("lead_time_days", 3))
        unit_price = float(med_info.get("unit_price", 10.0))

        optimizer = PharmacyInventoryOptimizer(default_service_level=args.service_level)
        policy = optimizer.compute_inventory_policy(
            predicted_daily_demand=avg_daily_demand,
            forecast_rmse=rmse,
            lead_time_days=lead_time,
            current_inventory=args.current_stock,
            service_level=args.service_level,
            unit_cost=unit_price
        )

        print(f"Medicine Name          : {med_info['medicine_name']} ({args.medicine})")
        print(f"Forecast Daily Demand  : {policy['predicted_daily_demand']} units/day")
        print(f"Supplier Lead Time     : {policy['lead_time_days']} days")
        print(f"Target Service Level   : {policy['service_level_pct']}%")
        print(f"Safety Stock Required  : {policy['safety_stock']} units")
        print(f"Reorder Point (ROP)    : {policy['reorder_point']} units")
        print(f"Current Inventory      : {policy['current_inventory']} units ({policy['days_of_supply']} days of supply)")
        print(f"Inventory Health Status: {policy['status']}")
        print(f"Stockout Risk in Cycle : {policy['stockout_risk_pct']}%")
        if policy["recommended_order_qty"] > 0:
            print(f"Recommended Order Qty  : {policy['recommended_order_qty']} units (EOQ: {policy['eoq']})")
        else:
            print("Action                 : Stock level adequate. No immediate reorder needed.")


if __name__ == "__main__":
    main()
