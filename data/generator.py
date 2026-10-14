"""
Realistic Pharmaceutical Sales and Demand Data Generator.

Generates multi-year, multi-medicine daily transaction and aggregated demand records
incorporating realistic healthcare patterns:
- Seasonal epidemiological peaks (winter flu, spring allergies, monsoon infections)
- Chronic prescription refill cycles (monthly peaks around days 1-5 and 28-31)
- Weekly clinic visitation patterns (Monday and Saturday peaks)
- Outbreak surges (viral/bacterial epidemic episodes)
- Promotions, price fluctuations, and supply shock/stockout markers
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


MEDICINE_CATALOG: List[Dict] = [
    {
        "medicine_id": "MED-001",
        "name": "Amoxicillin 500mg",
        "category": "Antibiotics",
        "prescription_required": True,
        "unit_price": 14.50,
        "base_demand": 45,
        "seasonal_type": "winter_respiratory",
        "refill_cyclic": False,
        "lead_time_days": 4,
        "shelf_life_days": 730,
    },
    {
        "medicine_id": "MED-002",
        "name": "Azithromycin 250mg",
        "category": "Antibiotics",
        "prescription_required": True,
        "unit_price": 22.00,
        "base_demand": 30,
        "seasonal_type": "winter_respiratory",
        "refill_cyclic": False,
        "lead_time_days": 5,
        "shelf_life_days": 730,
    },
    {
        "medicine_id": "MED-003",
        "name": "Paracetamol 650mg",
        "category": "Analgesics & Antipyretics",
        "prescription_required": False,
        "unit_price": 5.20,
        "base_demand": 120,
        "seasonal_type": "flu_and_monsoon",
        "refill_cyclic": False,
        "lead_time_days": 2,
        "shelf_life_days": 1095,
    },
    {
        "medicine_id": "MED-004",
        "name": "Ibuprofen 400mg",
        "category": "Analgesics & Antipyretics",
        "prescription_required": False,
        "unit_price": 8.00,
        "base_demand": 65,
        "seasonal_type": "steady",
        "refill_cyclic": False,
        "lead_time_days": 3,
        "shelf_life_days": 1095,
    },
    {
        "medicine_id": "MED-005",
        "name": "Cetirizine 10mg",
        "category": "Antihistamines",
        "prescription_required": False,
        "unit_price": 6.80,
        "base_demand": 50,
        "seasonal_type": "spring_fall_allergy",
        "refill_cyclic": False,
        "lead_time_days": 3,
        "shelf_life_days": 730,
    },
    {
        "medicine_id": "MED-006",
        "name": "Montelukast 10mg",
        "category": "Respiratory & Asthma",
        "prescription_required": True,
        "unit_price": 18.50,
        "base_demand": 40,
        "seasonal_type": "seasonal_asthma",
        "refill_cyclic": True,
        "lead_time_days": 4,
        "shelf_life_days": 730,
    },
    {
        "medicine_id": "MED-007",
        "name": "Metformin 500mg",
        "category": "Chronic - Diabetes",
        "prescription_required": True,
        "unit_price": 11.00,
        "base_demand": 95,
        "seasonal_type": "steady",
        "refill_cyclic": True,
        "lead_time_days": 5,
        "shelf_life_days": 1095,
    },
    {
        "medicine_id": "MED-008",
        "name": "Atorvastatin 20mg",
        "category": "Chronic - Cardiovascular",
        "prescription_required": True,
        "unit_price": 24.50,
        "base_demand": 75,
        "seasonal_type": "steady",
        "refill_cyclic": True,
        "lead_time_days": 4,
        "shelf_life_days": 1095,
    },
    {
        "medicine_id": "MED-009",
        "name": "Amlodipine 5mg",
        "category": "Chronic - Cardiovascular",
        "prescription_required": True,
        "unit_price": 9.50,
        "base_demand": 70,
        "seasonal_type": "steady",
        "refill_cyclic": True,
        "lead_time_days": 3,
        "shelf_life_days": 1095,
    },
    {
        "medicine_id": "MED-010",
        "name": "Omeprazole 20mg",
        "category": "Gastrointestinal",
        "prescription_required": False,
        "unit_price": 13.00,
        "base_demand": 55,
        "seasonal_type": "festive_gastric",
        "refill_cyclic": True,
        "lead_time_days": 3,
        "shelf_life_days": 730,
    },
]


def generate_pharmacy_sales(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
    medicines: Optional[List[Dict]] = None
) -> pd.DataFrame:
    """
    Generates realistic daily pharmacy sales records with seasonal cycles,
    weather influence, refill dynamics, and outbreak surges.
    """
    np.random.seed(seed)
    catalog = medicines if medicines is not None else MEDICINE_CATALOG
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)
    
    # Pre-generate macro factors (temperature, outbreak episodes, holidays)
    day_of_year = date_range.dayofyear.values
    
    # Approximate regional temperature curve (colder in Jan/Feb & Nov/Dec)
    temp_celsius = 25 - 12 * np.cos(2 * np.pi * (day_of_year - 15) / 365) + np.random.normal(0, 2, n_days)
    
    # Rainfall / monsoon seasonality (Jul-Aug)
    rainfall_prob = np.clip(0.6 * np.exp(-((day_of_year - 200) / 40) ** 2), 0.05, 0.9)
    rainfall_mm = np.where(np.random.rand(n_days) < rainfall_prob, np.random.gamma(2, 8, n_days), 0.0)
    
    # Outbreak surge simulation (e.g., flu waves, viral fever spikes)
    # 2 outbreaks per year lasting 14-25 days
    outbreak_multiplier = np.ones(n_days)
    for year in date_range.year.unique():
        for _ in range(2):
            surge_start_doy = np.random.choice([15, 45, 210, 310])
            surge_duration = np.random.randint(14, 28)
            surge_peak = np.random.uniform(1.6, 2.5)
            
            mask = (date_range.year == year) & (day_of_year >= surge_start_doy) & (day_of_year < surge_start_doy + surge_duration)
            if np.any(mask):
                curve = np.sin(np.linspace(0, np.pi, np.sum(mask)))
                outbreak_multiplier[mask] += (surge_peak - 1.0) * curve

    all_records = []

    for med in catalog:
        med_id = med["medicine_id"]
        name = med["name"]
        cat = med["category"]
        base = med["base_demand"]
        seasonal_type = med["seasonal_type"]
        is_cyclic = med["refill_cyclic"]
        unit_price = med["unit_price"]
        lead_time = med["lead_time_days"]
        shelf_life = med["shelf_life_days"]

        # 1. Long-term trend (gradual adoption or market shift: 3% - 8% annual growth)
        trend = np.linspace(1.0, np.random.uniform(1.08, 1.25), n_days)
        
        # 2. Seasonality component based on medicine class
        seasonality = np.ones(n_days)
        if seasonal_type == "winter_respiratory":
            # Peaks heavily during winter (cold weather)
            winter_factor = 1.0 + 0.65 * np.maximum(0, (20 - temp_celsius) / 15)
            seasonality *= winter_factor
        elif seasonal_type == "flu_and_monsoon":
            # Peaks during cold winter and wet monsoon
            flu_winter = 0.5 * np.maximum(0, (20 - temp_celsius) / 15)
            monsoon_bump = 0.4 * (rainfall_mm > 5).astype(float)
            seasonality *= (1.0 + flu_winter + monsoon_bump)
        elif seasonal_type == "spring_fall_allergy":
            # Peaks around DOY 80-120 (Spring) and DOY 250-290 (Autumn pollen)
            spring_peak = 0.8 * np.exp(-((day_of_year - 105) / 25) ** 2)
            autumn_peak = 0.5 * np.exp(-((day_of_year - 270) / 25) ** 2)
            seasonality *= (1.0 + spring_peak + autumn_peak)
        elif seasonal_type == "seasonal_asthma":
            cold_asthma = 0.4 * np.maximum(0, (18 - temp_celsius) / 15)
            seasonality *= (1.0 + cold_asthma)
        elif seasonal_type == "festive_gastric":
            # Year-end and festive periods (Nov-Dec, DOY 300-365)
            festive_bump = 0.45 * np.exp(-((day_of_year - 335) / 30) ** 2)
            seasonality *= (1.0 + festive_bump)

        # Apply outbreak factor mainly to antibiotics, analgesics, antipyretics
        if seasonal_type in ["winter_respiratory", "flu_and_monsoon"]:
            seasonality *= outbreak_multiplier

        # 3. Day of week pattern:
        # Mon (0) is busiest, Sat (5) busy, Sun (6) quieter
        dow_multipliers = {
            0: 1.25,  # Monday doctor clinic rush
            1: 1.05,
            2: 1.00,
            3: 0.98,
            4: 1.10,
            5: 1.18,  # Saturday weekend restocking
            6: 0.65   # Sunday half-day / low clinic attendance
        }
        dow_factor = np.array([dow_multipliers[d] for d in date_range.dayofweek])

        # 4. Monthly refill cycle for chronic medicines (days 1-5 and 27-31)
        refill_factor = np.ones(n_days)
        if is_cyclic:
            dom = date_range.day.values
            month_start_mask = (dom <= 5)
            month_end_mask = (dom >= 27)
            refill_factor[month_start_mask] += 0.35
            refill_factor[month_end_mask] += 0.25

        # 5. Promotions and discounts (10% to 15% discount campaigns scheduled occasionally)
        is_promo = np.zeros(n_days, dtype=int)
        # 4 promotional campaigns per year of 4-7 days
        for year in date_range.year.unique():
            campaign_starts = np.random.choice(date_range[date_range.year == year].dayofyear, size=4, replace=False)
            for c_start in campaign_starts:
                c_dur = np.random.randint(3, 7)
                c_mask = (date_range.year == year) & (day_of_year >= c_start) & (day_of_year < c_start + c_dur)
                is_promo[c_mask] = 1

        promo_lift = 1.0 + is_promo * 0.28  # 28% demand lift during promotions
        actual_price = np.where(is_promo == 1, np.round(unit_price * 0.88, 2), unit_price)

        # 6. Synthesize mean demand and apply Poisson/Negative-Binomial random noise
        mean_units = base * trend * seasonality * dow_factor * refill_factor * promo_lift
        
        # Add realistic random dispersion
        noise = np.random.normal(1.0, 0.12, n_days)
        projected_demand = np.maximum(0, mean_units * noise)
        
        # Round to whole integer units sold
        units_sold = np.random.poisson(projected_demand).astype(int)

        # 7. Stockout simulation (rare inventory stockout occurrences, 1-2 events/year)
        stockout_flag = np.zeros(n_days, dtype=int)
        for year in date_range.year.unique():
            if np.random.rand() > 0.3:
                so_start = np.random.choice(date_range[date_range.year == year].dayofyear)
                so_dur = np.random.randint(1, 4)
                so_mask = (date_range.year == year) & (day_of_year >= so_start) & (day_of_year < so_start + so_dur)
                stockout_flag[so_mask] = 1
                # On stockout days, sales drop drastically
                units_sold[so_mask] = np.random.randint(0, 4, size=np.sum(so_mask))

        revenue = np.round(units_sold * actual_price, 2)

        for i in range(n_days):
            all_records.append({
                "date": date_range[i].strftime("%Y-%m-%d"),
                "medicine_id": med_id,
                "medicine_name": name,
                "category": cat,
                "units_sold": int(units_sold[i]),
                "unit_price": float(actual_price[i]),
                "revenue": float(revenue[i]),
                "is_promotional": int(is_promo[i]),
                "stockout_flag": int(stockout_flag[i]),
                "lead_time_days": lead_time,
                "temp_celsius": round(float(temp_celsius[i]), 1),
                "rainfall_mm": round(float(rainfall_mm[i]), 1),
                "is_outbreak_period": int(outbreak_multiplier[i] > 1.25),
            })

    df = pd.DataFrame(all_records)
    return df


def save_dataset_if_not_exists(
    filepath: str = "data/raw_pharmacy_sales.csv",
    force_regenerate: bool = False
) -> str:
    """Creates and saves the dataset if it does not already exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath) and not force_regenerate:
        return filepath
    
    print(f"Generating realistic pharmacy dataset at {filepath}...")
    df = generate_pharmacy_sales()
    df.to_csv(filepath, index=False)
    print(f"Successfully generated {len(df)} records across {df['medicine_id'].nunique()} medicines.")
    return filepath


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(current_dir, "raw_pharmacy_sales.csv")
    save_dataset_if_not_exists(target_path, force_regenerate=True)
