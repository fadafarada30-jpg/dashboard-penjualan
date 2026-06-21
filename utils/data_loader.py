"""
data_loader.py
Modul untuk load, cleaning, dan feature engineering dataset Superstore.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str = "data/Sample_-_Superstore.csv") -> pd.DataFrame:
    """
    Load dataset Superstore dengan encoding yang sesuai (Latin-1)
    dan lakukan cleaning dasar.
    """
    df = pd.read_csv(filepath, encoding="latin-1")

    # Standarisasi nama kolom: lowercase + underscore
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Parsing tanggal
    df["order_date"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y", errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], format="%m/%d/%Y", errors="coerce")

    # Drop duplikat kalau ada
    df = df.drop_duplicates()

    # Pastikan tipe numerik benar
    num_cols = ["sales", "quantity", "discount", "profit"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop baris yang order_date-nya gagal di-parse (data rusak)
    df = df.dropna(subset=["order_date"])

    # Feature tambahan yang berguna buat analisis
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["profit_margin"] = np.where(df["sales"] != 0, df["profit"] / df["sales"], 0)

    return df.reset_index(drop=True)


def get_kpi_summary(df: pd.DataFrame) -> dict:
    """Hitung KPI ringkas untuk ditampilkan di dashboard."""
    return {
        "total_sales": df["sales"].sum(),
        "total_profit": df["profit"].sum(),
        "total_orders": df["order_id"].nunique(),
        "total_customers": df["customer_id"].nunique(),
        "avg_order_value": df.groupby("order_id")["sales"].sum().mean(),
        "profit_margin": df["profit"].sum() / df["sales"].sum() if df["sales"].sum() else 0,
    }


if __name__ == "__main__":
    # Quick test
    df = load_data()
    print(f"Jumlah baris setelah cleaning: {len(df)}")
    print(f"Rentang tanggal: {df['order_date'].min()} s/d {df['order_date'].max()}")
    print(f"Kolom: {list(df.columns)}")
    print("\nKPI Summary:")
    for k, v in get_kpi_summary(df).items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")