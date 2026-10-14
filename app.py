"""
Pharmacy Demand Forecaster - Modern Healthcare AI Dashboard.
Predicts medicine sales using past demand and seasonal patterns with lag features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# Ensure project modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_and_validate_data
from data.generator import save_dataset_if_not_exists, MEDICINE_CATALOG
from features.engineer import PharmacyFeatureEngineer
from models.forecaster import PharmacyDemandForecaster, benchmark_all_models, MODEL_REGISTRY
from models.inventory_optimizer import PharmacyInventoryOptimizer


# Configure Matplotlib default aesthetic
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#F1F5F9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PharmAI - Demand Forecaster & Inventory Optimizer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-banner h1 {
        color: #F8FAFC !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.025em;
    }
    .hero-banner p {
        color: #CBD5E1 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
        font-weight: 400;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(45, 212, 191, 0.2);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }

    /* Medicine Info Card */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        justify-content: space-between;
        align-items: center;
    }
    .med-card .med-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .med-card .med-meta {
        font-size: 0.88rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* KPI Summary Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0,0,0,0.06);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        font-weight: 500;
        color: #059669;
    }

    /* Status Badges */
    .status-badge-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-badge-warning {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FCD34D;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-badge-healthy {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #86EFAC;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_default_data():
    raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_pharmacy_sales.csv")
    if not os.path.exists(raw_path):
        save_dataset_if_not_exists(raw_path, force_regenerate=True)
    df, meta = load_and_validate_data(raw_path)
    return df, meta


def main():
    # Hero Header Banner
    st.markdown("""
    <div class="hero-banner">
        <span class="badge-pill">Clinical ML Intelligence • Regression with Lag Dynamics</span>
        <h1>💊 Pharmacy Demand Forecaster</h1>
        <p>Predicts medication demand, seasonal epidemiological surges, and optimizes inventory reorder points to prevent stockouts.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration & Controls
    st.sidebar.markdown("### ⚙️ Control Panel")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["🏥 Clinical Reference Catalog (10 Medicines)", "📁 Upload Custom Pharmacy CSV"],
        index=0
    )

    if data_source == "📁 Upload Custom Pharmacy CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Pharmacy Sales CSV",
            type=["csv"],
            help="Expected columns: date, medicine_id, units_sold. Optional: medicine_name, category, unit_price, lead_time_days."
        )
        if uploaded_file is not None:
            try:
                df, metadata = load_and_validate_data(uploaded_file)
                st.sidebar.success(f"✓ Loaded {len(df):,} records across {metadata['num_medicines']} medicines!")
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {e}")
                return
        else:
            st.info("Upload a CSV file to proceed, or choose the Clinical Reference Catalog.")
            return
    else:
        df, metadata = get_default_data()

    # Category Quick Filter
    categories = ["All Categories"] + sorted(df["category"].dropna().unique().tolist())
    selected_cat = st.sidebar.selectbox("Filter by Category", categories, index=0)

    # Filtered Medicine Dropdown
    if selected_cat != "All Categories":
        filtered_df = df[df["category"] == selected_cat]
    else:
        filtered_df = df

    med_unique = filtered_df[["medicine_id", "medicine_name", "category"]].drop_duplicates().to_dict(orient="records")
    med_options = {f"{m['medicine_name']} ({m['medicine_id']})": m["medicine_id"] for m in med_unique}

    if not med_options:
        st.warning("No medicines found for selected category.")
        return

    selected_med_label = st.sidebar.selectbox("Select Medication", list(med_options.keys()), index=0)
    selected_med_id = med_options[selected_med_label]
    selected_med_info = df[df["medicine_id"] == selected_med_id].iloc[-1]

    # Model Hyperparameter Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Machine Learning Model")
    selected_model = st.sidebar.selectbox(
        "Forecasting Model",
        ["xgboost", "random_forest", "gradient_boosting", "ridge"],
        format_func=lambda x: {
            "xgboost": "XGBoost Regressor (Gradient Boosting)",
            "random_forest": "Random Forest Regressor (Ensemble)",
            "gradient_boosting": "Gradient Boosting (Hist/Trees)",
            "ridge": "Ridge Linear Regression (L2 Baseline)"
        }[x],
        index=0
    )

    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=14, step=7)
    service_level = st.sidebar.select_slider(
        "Target Service Level (Fulfillment)",
        options=[0.90, 0.95, 0.98, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x * 100)}% (Critical Healthcare Standard)" if x >= 0.95 else f"{int(x * 100)}%"
    )

    # Train model on selected medicine
    @st.cache_resource(show_spinner=False)
    def train_cached_forecaster(data_len: int, total_units: int, model_name: str, med_id: str):
        forecaster = PharmacyDemandForecaster(model_name=model_name, test_days=60)
        train_output = forecaster.train(df, medicine_id=med_id)
        return forecaster, train_output

    data_hash = hash(len(df)) + hash(int(df["units_sold"].sum()))
    forecaster, train_output = train_cached_forecaster(len(df), int(df["units_sold"].sum()), selected_model, selected_med_id)
    metrics = train_output["metrics"]

    # Active Medicine Profile Header
    rx_badge = "🛡️ Prescription Required (Rx)" if selected_med_info.get("prescription_required", True) else "💊 Over-The-Counter (OTC)"
    st.markdown(f"""
    <div class="med-card">
        <div>
            <h3 class="med-title">{selected_med_info.get('medicine_name', selected_med_id)}</h3>
            <div class="med-meta">
                ID: <b>{selected_med_id}</b> &nbsp;|&nbsp; 
                Category: <b>{selected_med_info.get('category', 'General')}</b> &nbsp;|&nbsp; 
                {rx_badge}
            </div>
        </div>
        <div style="display:flex; gap:20px; text-align:right;">
            <div>
                <div style="font-size:0.78rem; color:#64748B; font-weight:600; text-transform:uppercase;">Unit Price</div>
                <div style="font-size:1.25rem; font-weight:700; color:#0F172A;">${float(selected_med_info.get('unit_price', 10.0)):.2f}</div>
            </div>
            <div>
                <div style="font-size:0.78rem; color:#64748B; font-weight:600; text-transform:uppercase;">Lead Time</div>
                <div style="font-size:1.25rem; font-weight:700; color:#0F172A;">{int(selected_med_info.get('lead_time_days', 3))} Days</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Row
    med_df = df[df["medicine_id"] == selected_med_id].sort_values("date").reset_index(drop=True)
    total_historical = int(med_df["units_sold"].sum())
    daily_avg = float(med_df["units_sold"].mean())
    last_30_avg = float(med_df["units_sold"].tail(30).mean())
    growth_30d = ((last_30_avg - daily_avg) / max(0.1, daily_avg)) * 100

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box">
            <div class="kpi-label">Lifetime Demand</div>
            <div class="kpi-value">{total_historical:,}</div>
            <div class="kpi-sub">Recorded Patient Units</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Historical Daily Velocity</div>
            <div class="kpi-value">{daily_avg:.1f}</div>
            <div class="kpi-sub">Units / Day Baseline</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Recent 30-Day Run Rate</div>
            <div class="kpi-value">{last_30_avg:.1f}</div>
            <div class="kpi-sub" style="color:{'#059669' if growth_30d >= 0 else '#DC2626'};">{growth_30d:+.1f}% vs Lifetime Avg</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Holdout Model Accuracy</div>
            <div class="kpi-value">{100.0 - metrics['wape']:.1f}%</div>
            <div class="kpi-sub">WAPE: {metrics['wape']:.1f}% | RMSE: {metrics['rmse']:.1f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Tabs
    tab_forecast, tab_inv, tab_sim, tab_explorer, tab_bench = st.tabs([
        "🔮 Future Demand Forecast",
        "📦 Inventory & Reorder Intelligence",
        "🧪 What-If Clinical Simulator",
        "📈 Demand Analytics & Seasonality",
        "⚖️ Model Benchmark & ML Diagnostics"
    ])

    # ------------------------------------------------------------------
    # TAB 1: Future Demand Forecast
    # ------------------------------------------------------------------
    with tab_forecast:
        st.subheader(f"Next {forecast_horizon}-Day Demand Trajectory")
        
        forecast_df = forecaster.forecast_future(
            df_history=df,
            medicine_id=selected_med_id,
            horizon_days=forecast_horizon,
            confidence_level=0.95
        )

        total_projected = float(forecast_df["predicted_demand"].sum())
        avg_projected = float(forecast_df["predicted_demand"].mean())
        unit_price = float(selected_med_info.get("unit_price", 10.0))
        projected_rev = total_projected * unit_price

        # High-impact summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Total Projected Units ({forecast_horizon}d)", f"{int(total_projected):,} units")
        c2.metric("Projected Daily Rate", f"{avg_projected:.1f} units/day")
        c3.metric("Estimated Revenue", f"${projected_rev:,.2f}")
        c4.metric("Forecast Confidence Interval", "95% Upper / Lower Band")

        st.write("")
        # Forecast Visual Chart
        tail_history = med_df.tail(45).copy()
        future_dates = pd.to_datetime(forecast_df["date"])

        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
        ax.plot(tail_history["date"], tail_history["units_sold"], label="Historical Actual Sales", color="#475569", linewidth=1.6)
        ax.plot(future_dates, forecast_df["predicted_demand"], label=f"Predicted Demand ({selected_model.upper()})", color="#2563EB", linewidth=2.5, marker="o", markersize=4.5)
        ax.fill_between(future_dates, forecast_df["lower_ci"], forecast_df["upper_ci"], color="#60A5FA", alpha=0.3, label="95% Prediction Interval")
        ax.axvline(x=tail_history["date"].iloc[-1], color="#DC2626", linestyle="--", linewidth=1.5, label="Today (Forecast Start)")

        ax.set_ylabel("Daily Units Sold", fontsize=11, fontweight=600)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Forecast Breakdown & Export
        st.markdown("#### 📅 Daily Projection Schedule")
        col_table, col_actions = st.columns([3, 1])
        with col_table:
            st.dataframe(
                forecast_df.rename(columns={
                    "date": "Date",
                    "day_name": "Day",
                    "predicted_demand": "Predicted Units",
                    "lower_ci": "Lower 95% CI",
                    "upper_ci": "Upper 95% CI"
                }),
                use_container_width=True,
                height=250
            )
        with col_actions:
            st.markdown("##### 📥 Export Data")
            csv_data = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Forecast CSV",
                data=csv_data,
                file_name=f"pharmacy_forecast_{selected_med_id}_{forecast_horizon}d.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption(f"Includes daily point estimates and confidence intervals for {forecast_horizon} days.")

    # ------------------------------------------------------------------
    # TAB 2: Inventory & Reorder Intelligence
    # ------------------------------------------------------------------
    with tab_inv:
        st.subheader("Automated Pharmacy Inventory & Safety Stock Engine")
        st.caption("Translates forecast mean and residual volatility into buffer stock & reorder limits to prevent stockouts.")

        # Interactive On-Hand Inventory Slider
        inv_ctrl_col1, inv_ctrl_col2 = st.columns([2, 1])
        with inv_ctrl_col1:
            stock_input = st.slider(
                "Adjust Current On-Hand Stock (Units)",
                min_value=0,
                max_value=1200,
                value=st.session_state.get("custom_stock", 140),
                step=10,
                help="Slide to simulate different stock levels and watch the automated replenishment engine respond."
            )
        with inv_ctrl_col2:
            lead_time_override = st.number_input(
                "Supplier Lead Time (Days)",
                min_value=1,
                max_value=30,
                value=int(selected_med_info.get("lead_time_days", 3))
            )

        # Compute Inventory Policy
        forecast_14d = forecaster.forecast_future(df, medicine_id=selected_med_id, horizon_days=14)
        avg_future_demand = float(forecast_14d["predicted_demand"].mean())
        optimizer = PharmacyInventoryOptimizer(default_service_level=service_level)
        policy = optimizer.compute_inventory_policy(
            predicted_daily_demand=avg_future_demand,
            forecast_rmse=metrics["rmse"],
            lead_time_days=lead_time_override,
            current_inventory=stock_input,
            service_level=service_level,
            unit_cost=float(selected_med_info.get("unit_price", 10.0))
        )

        # Status Alert Badge
        status = policy["status"]
        if status == "CRITICAL_STOCKOUT_RISK":
            st.markdown(f"""
            <div class="status-badge-critical">
                <span style="font-size:1.3rem;">🚨</span>
                <div>
                    <b>CRITICAL STOCKOUT ALERT:</b> On-hand stock ({stock_input} units) is lower than the required Safety Stock buffer ({policy['safety_stock']} units)! Place emergency replenishment immediately.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif status == "REORDER_TRIGGERED":
            st.markdown(f"""
            <div class="status-badge-warning">
                <span style="font-size:1.3rem;">⚡</span>
                <div>
                    <b>REORDER POINT BREACHED:</b> Current inventory ({stock_input} units) is at or below the Reorder Point ({policy['reorder_point']} units). Standard replenishment order triggered.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-badge-healthy">
                <span style="font-size:1.3rem;">✅</span>
                <div>
                    <b>INVENTORY HEALTHY:</b> Stock level ({stock_input} units) provides {policy['days_of_supply']} days of supply. Stockout risk during supplier lead time is minimal ({policy['stockout_risk_pct']}%).
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Policy KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Safety Stock Buffer", f"{policy['safety_stock']} units")
        k2.metric("Reorder Point (ROP)", f"{policy['reorder_point']} units")
        k3.metric("Days of Supply Remaining", f"{policy['days_of_supply']} days")
        k4.metric("Stockout Risk in Cycle", f"{policy['stockout_risk_pct']}%")

        st.markdown("---")
        # Visual Stock Level Gauge Bar
        st.write("#### 📊 Stock Level vs. Buffer Thresholds")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 2.2), dpi=120)
        
        max_limit = max(stock_input, policy["reorder_point"] * 1.6)
        ax_bar.barh(["Inventory Units"], [stock_input], color="#2563EB", height=0.45, label=f"Current Stock ({stock_input})", zorder=3)
        ax_bar.axvline(x=policy["safety_stock"], color="#DC2626", linestyle="--", linewidth=2, label=f"Safety Stock ({policy['safety_stock']})")
        ax_bar.axvline(x=policy["reorder_point"], color="#D97706", linestyle="-.", linewidth=2, label=f"Reorder Point ({policy['reorder_point']})")
        ax_bar.set_xlim(0, max_limit * 1.1)
        ax_bar.set_xlabel("Units", fontweight=600)
        ax_bar.legend(loc="upper right", frameon=True)
        ax_bar.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Recommendation Actions
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("##### 📐 Calculation Breakdown")
            st.markdown(f"""
            - **Forecasted Demand During Lead Time**: `{policy['predicted_daily_demand']} units/day × {policy['lead_time_days']} days = {round(policy['predicted_daily_demand'] * policy['lead_time_days'], 1)} units`
            - **Safety Buffer (Z = {int(service_level*100)}%)**: `{policy['safety_stock']} units`
            - **Reorder Point (ROP)**: `{round(policy['predicted_daily_demand'] * policy['lead_time_days'], 1)} + {policy['safety_stock']} = {policy['reorder_point']} units`
            """)
        with p_col2:
            st.markdown("##### 🛒 Recommended Order Action")
            if policy["recommended_order_qty"] > 0:
                cost_estimate = policy["recommended_order_qty"] * float(selected_med_info.get("unit_price", 10.0))
                st.success(f"""
                **Order Recommendation**: Purchase **{policy['recommended_order_qty']} units**
                - Optimal Batch (EOQ): **{policy['eoq']} units**
                - Estimated Purchase Cost: **${cost_estimate:,.2f}**
                """)
                if st.button("Simulate Restock (+Recommended Units)"):
                    st.session_state["custom_stock"] = stock_input + policy["recommended_order_qty"]
                    st.rerun()
            else:
                st.info("Stock levels are currently optimal. No immediate purchase order required.")

    # ------------------------------------------------------------------
    # TAB 3: What-If Clinical Simulator
    # ------------------------------------------------------------------
    with tab_sim:
        st.subheader("🧪 'What-If' Epidemiological & Campaign Stress Simulator")
        st.caption("Stress-test the supply chain against unexpected disease waves, promotional discounts, or supplier delays.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            outbreak_surge = st.slider("🦠 Epidemic / Outbreak Surge Intensity", min_value=0, max_value=150, value=50, step=10, format="+%d%%")
        with s_col2:
            promo_discount = st.slider("🏷️ Promotional Campaign Discount", min_value=0, max_value=30, value=10, step=5, format="%d%%")

        # Run Stressed Forecast
        stressed_factor = 1.0 + (outbreak_surge / 100.0) + (promo_discount * 0.015)
        base_forecast = forecaster.forecast_future(df, medicine_id=selected_med_id, horizon_days=14)
        stressed_demand = base_forecast["predicted_demand"] * stressed_factor

        base_total = base_forecast["predicted_demand"].sum()
        stressed_total = stressed_demand.sum()
        delta_units = stressed_total - base_total

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Baseline 14-Day Demand", f"{int(base_total):,} units")
        sc2.metric("Stressed 14-Day Demand", f"{int(stressed_total):,} units", delta=f"+{delta_units:.0f} units")
        surge_rop = int(np.ceil((stressed_demand.mean() * int(selected_med_info.get("lead_time_days", 3))) + (metrics["rmse"] * 1.645 * np.sqrt(3))))
        sc3.metric("Adjusted Reorder Point", f"{surge_rop} units", delta=f"+{surge_rop - policy['reorder_point']} units")

        # Visual Comparison Chart
        fig_sim, ax_sim = plt.subplots(figsize=(11, 4.2), dpi=120)
        dates_sim = pd.to_datetime(base_forecast["date"])
        ax_sim.plot(dates_sim, base_forecast["predicted_demand"], label="Baseline Forecast", color="#64748B", linestyle="--", linewidth=2)
        ax_sim.plot(dates_sim, stressed_demand, label=f"Stressed Scenario (+{outbreak_surge}% Outbreak, {promo_discount}% Promo)", color="#DC2626", linewidth=2.5, marker="o")
        ax_sim.fill_between(dates_sim, base_forecast["predicted_demand"], stressed_demand, color="#FCA5A5", alpha=0.3, label="Surge Gap (Deficit Risk)")
        ax_sim.set_ylabel("Daily Units Sold", fontweight=600)
        ax_sim.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_sim.legend(loc="upper left", frameon=True)
        ax_sim.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_sim)
        plt.close(fig_sim)

    # ------------------------------------------------------------------
    # TAB 4: Demand Analytics & Seasonality
    # ------------------------------------------------------------------
    with tab_explorer:
        st.subheader("Historical Demand Analytics & Seasonal Cycles")
        
        # Historical Trend Line
        fig_hist, ax_hist = plt.subplots(figsize=(12, 4.2), dpi=120)
        ax_hist.plot(med_df["date"], med_df["units_sold"], label="Daily Units Sold", color="#94A3B8", alpha=0.5, linewidth=1)
        ax_hist.plot(med_df["date"], med_df["units_sold"].rolling(7).mean(), label="7-Day Moving Avg", color="#2563EB", linewidth=2)
        ax_hist.plot(med_df["date"], med_df["units_sold"].rolling(28).mean(), label="28-Day Moving Avg", color="#DC2626", linewidth=1.8, linestyle="--")
        
        if "is_outbreak_period" in med_df.columns and med_df["is_outbreak_period"].sum() > 0:
            outbreak_pts = med_df[med_df["is_outbreak_period"] == 1]
            ax_hist.scatter(outbreak_pts["date"], outbreak_pts["units_sold"], color="#E11D48", s=18, alpha=0.5, label="Outbreak Episodes", zorder=4)

        ax_hist.set_ylabel("Units Sold", fontweight=600)
        ax_hist.legend(loc="upper left", frameon=True)
        ax_hist.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # Weekly and Monthly Heatmaps
        w_col, m_col = st.columns(2)
        with w_col:
            st.markdown("##### 📅 Day-of-Week Visitation Pattern")
            med_df["day_name"] = med_df["date"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_data = med_df.groupby("day_name")["units_sold"].mean().reindex(day_order)
            
            fig_dow, ax_dow = plt.subplots(figsize=(6, 3.2), dpi=110)
            colors = ["#2563EB" if d in ["Monday", "Saturday"] else "#64748B" for d in day_order]
            ax_dow.bar(dow_data.index, dow_data.values, color=colors)
            ax_dow.set_ylabel("Avg Units", fontweight=600)
            ax_dow.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            ax_dow.grid(axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_dow)
            plt.close(fig_dow)

        with m_col:
            st.markdown("##### 🌦️ Annual Monthly Seasonality")
            monthly_data = med_df.groupby(med_df["date"].dt.month)["units_sold"].mean()
            fig_m, ax_m = plt.subplots(figsize=(6, 3.2), dpi=110)
            ax_m.plot(monthly_data.index, monthly_data.values, marker="s", color="#0D9488", linewidth=2.2)
            ax_m.set_xticks(range(1, 13))
            ax_m.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
            ax_m.set_ylabel("Avg Units", fontweight=600)
            ax_m.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

    # ------------------------------------------------------------------
    # TAB 5: Model Benchmark & ML Diagnostics
    # ------------------------------------------------------------------
    with tab_bench:
        st.subheader("Model Benchmarking & Feature Attribution")
        st.caption("Side-by-side evaluation across 4 regression algorithms evaluated on the identical 60-day holdout test split.")

        bm_col, metric_col = st.columns([3, 2])
        with bm_col:
            with st.spinner("Benchmarking all models..."):
                bench_df = benchmark_all_models(df, medicine_id=selected_med_id, test_days=60)
            st.dataframe(bench_df, use_container_width=True)

        with metric_col:
            st.markdown("##### 🎯 Active Model Performance Card")
            st.json({
                "Algorithm": selected_model.upper(),
                "RMSE (Root Mean Sq Error)": f"{metrics['rmse']} units",
                "MAE (Mean Absolute Error)": f"{metrics['mae']} units",
                "WAPE (Weighted Abs Error)": f"{metrics['wape']}%",
                "R² Score (Explained Var)": f"{metrics['r2']}",
                "Tracking Bias": f"{metrics['bias']:+} units"
            })

        st.markdown("---")
        # Feature Importance Chart
        st.markdown("#### 🔍 Feature Importances (Lag & Seasonal Drivers)")
        feat_df = train_output["feature_importances"]
        if not feat_df.empty:
            top_feats = feat_df.head(10).sort_values("importance", ascending=True)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 4.2), dpi=120)
            ax_fi.barh(top_feats["feature"], top_feats["importance"], color="#4F46E5", height=0.6)
            ax_fi.set_xlabel("Relative Feature Importance Weight", fontweight=600)
            ax_fi.grid(axis="x", linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close(fig_fi)

        # Holdout Actual vs Predicted
        with st.expander("📈 View Holdout Set Backtest (Actuals vs Predictions over Last 60 Days)"):
            test_dates = pd.to_datetime(train_output["test_dates"])
            fig_test, ax_test = plt.subplots(figsize=(11, 3.8), dpi=120)
            ax_test.plot(test_dates, train_output["test_actuals"], label="Actual Sales (Test Set)", color="#059669", linewidth=2)
            ax_test.plot(test_dates, train_output["test_predictions"], label="Model Prediction", color="#D97706", linestyle="--", linewidth=2)
            ax_test.set_ylabel("Units Sold")
            ax_test.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax_test.legend(loc="upper left")
            ax_test.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_test)
            plt.close(fig_test)


if __name__ == "__main__":
    main()
