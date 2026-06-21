"""
analysis.py
Modul algoritma analisis: RFM Segmentation, K-Means Clustering,
dan Sales Forecasting sederhana.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ---------------------------------------------------------
# 1. RFM ANALYSIS (Recency, Frequency, Monetary)
# ---------------------------------------------------------
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung skor RFM per customer.
    - Recency  : berapa hari sejak transaksi terakhir
    - Frequency: jumlah order unik
    - Monetary : total sales yang dihasilkan
    """
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        customer_name=("customer_name", "first"),
        recency=("order_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("sales", "sum"),
    ).reset_index()

    # Scoring 1-5 pakai quantile (5 = paling bagus)
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label_segment(score):
        if score >= 13:
            return "Champion"
        elif score >= 10:
            return "Loyal Customer"
        elif score >= 7:
            return "Potential"
        elif score >= 4:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment_label"] = rfm["rfm_score"].apply(label_segment)
    return rfm


# ---------------------------------------------------------
# 2. K-MEANS CLUSTERING (segmentasi otomatis berbasis RFM)
# ---------------------------------------------------------
def kmeans_segmentation(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster customer berdasarkan Recency, Frequency, Monetary
    menggunakan algoritma K-Means.
    """
    features = rfm[["recency", "frequency", "monetary"]].copy()

    # Standarisasi fitur (penting karena skala R/F/M beda jauh)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = kmeans.fit_predict(features_scaled)

    # Beri nama cluster berdasarkan rata-rata monetary tiap cluster
    cluster_rank = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False)
    rank_map = {cluster_id: i for i, cluster_id in enumerate(cluster_rank.index)}
    cluster_names = ["High Value", "Medium-High Value", "Medium-Low Value", "Low Value"]
    name_map = {cid: cluster_names[rank] for cid, rank in rank_map.items() if rank < len(cluster_names)}
    rfm["cluster_label"] = rfm["cluster"].map(name_map).fillna("Other")

    return rfm


# ---------------------------------------------------------
# 3. SALES FORECASTING
#    Default: Holt-Winters Exponential Smoothing (trend + seasonality)
#    Fallback: Linear Regression (kalau statsmodels gak ada / data terlalu pendek)
# ---------------------------------------------------------
def forecast_sales(df: pd.DataFrame, periods_ahead: int = 3, method: str = "auto") -> pd.DataFrame:
    """
    Prediksi total sales bulan-bulan ke depan.

    method:
      - "holt_winters" : Triple Exponential Smoothing, menangkap tren + seasonality
                          tahunan. Butuh minimal ~24 bulan data historis.
      - "linear"        : Regresi linear sederhana atas tren waktu.
      - "auto"          : pakai Holt-Winters kalau memungkinkan, kalau tidak fallback
                          ke Linear Regression.
    """
    monthly = (
        df.groupby("order_year_month")["sales"]
        .sum()
        .reset_index()
        .sort_values("order_year_month")
        .reset_index(drop=True)
    )
    monthly["t"] = np.arange(len(monthly))

    use_hw = STATSMODELS_AVAILABLE and len(monthly) >= 24 and method in ("auto", "holt_winters")

    last_period = pd.Period(monthly["order_year_month"].iloc[-1])
    future_periods = [str(last_period + i) for i in range(1, periods_ahead + 1)]

    if use_hw:
        model_used = "Holt-Winters (trend + seasonal)"
        series = pd.Series(monthly["sales"].values, index=pd.period_range(
            start=monthly["order_year_month"].iloc[0], periods=len(monthly), freq="M"
        ))
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal="add", seasonal_periods=12,
                initialization_method="estimated"
            ).fit()
            future_pred = model.forecast(periods_ahead).values
        except Exception:
            # Data kurang stabil untuk seasonal model -> fallback
            use_hw = False

    if not use_hw:
        model_used = "Linear Regression (fallback)"
        X = monthly[["t"]]
        y = monthly["sales"]
        lr = LinearRegression().fit(X, y)
        future_t = np.arange(len(monthly), len(monthly) + periods_ahead).reshape(-1, 1)
        future_pred = lr.predict(future_t)

    future_pred = np.clip(future_pred, a_min=0, a_max=None)  # sales gak mungkin negatif

    forecast_df = pd.DataFrame({
        "order_year_month": future_periods,
        "sales": future_pred,
        "type": "forecast"
    })
    monthly["type"] = "actual"

    result = pd.concat(
        [monthly[["order_year_month", "sales", "type"]], forecast_df], ignore_index=True
    )
    result.attrs["model_used"] = model_used
    return result


# ---------------------------------------------------------
# 4. TOP PRODUCTS RANKING (sorting klasik)
# ---------------------------------------------------------
def top_products(df: pd.DataFrame, by: str = "sales", n: int = 10) -> pd.DataFrame:
    """
    Ranking produk terlaris berdasarkan metrik tertentu (sales/profit/quantity).
    Menggunakan sorting (merge sort di balik layar via pandas .sort_values).
    """
    summary = df.groupby("product_name").agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        total_quantity=("quantity", "sum"),
        category=("category", "first"),
    ).reset_index()

    col_map = {"sales": "total_sales", "profit": "total_profit", "quantity": "total_quantity"}
    sort_col = col_map.get(by, "total_sales")

    return summary.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------
# 5. MARKET BASKET ANALYSIS sederhana (produk sering dibeli bareng)
# ---------------------------------------------------------
def frequently_bought_together(df: pd.DataFrame, min_orders: int = 2, top_n: int = 10) -> pd.DataFrame:
    """
    Cari pasangan sub-kategori produk yang sering muncul di order yang sama.
    Pendekatan sederhana (tanpa library mlxtend) berbasis co-occurrence per order_id.
    """
    from itertools import combinations
    from collections import Counter

    order_groups = df.groupby("order_id")["sub_category"].apply(lambda x: sorted(set(x)))
    pair_counter = Counter()

    for items in order_groups:
        if len(items) > 1:
            for pair in combinations(items, 2):
                pair_counter[pair] += 1

    pairs_df = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counter.items() if c >= min_orders],
        columns=["product_a", "product_b", "co_occurrence"]
    )
    return pairs_df.sort_values("co_occurrence", ascending=False).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------
# 6. GEOGRAPHIC ANALYSIS (sales per state)
# ---------------------------------------------------------
US_STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def sales_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi sales & profit per state, lengkap dengan kode abbreviation
    untuk dipakai di choropleth map.
    """
    state_summary = df.groupby("state").agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        total_orders=("order_id", "nunique"),
    ).reset_index()

    state_summary["state_code"] = state_summary["state"].map(US_STATE_ABBREV)
    state_summary["profit_margin"] = state_summary["total_profit"] / state_summary["total_sales"]
    return state_summary.dropna(subset=["state_code"]).sort_values("total_sales", ascending=False)


# ---------------------------------------------------------
# 7. DISCOUNT IMPACT ANALYSIS
#    Membantu seller memahami: diskon di level berapa yang masih "aman"
#    sebelum mulai menggerus profit.
# ---------------------------------------------------------
def discount_impact_analysis(df: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """
    Bagi data jadi beberapa bucket discount, lalu hitung rata-rata
    profit margin & total sales per bucket.
    """
    data = df.copy()
    data["discount_bucket"] = pd.cut(
        data["discount"], bins=np.linspace(0, max(data["discount"].max(), 0.01), bins + 1),
        include_lowest=True
    )
    data["discount_bucket_label"] = data["discount_bucket"].apply(
        lambda x: f"{max(x.left, 0)*100:.0f}%-{x.right*100:.0f}%" if pd.notna(x) else "N/A"
    )

    summary = data.groupby("discount_bucket_label", observed=True).agg(
        avg_profit_margin=("profit_margin", "mean"),
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        order_count=("order_id", "nunique"),
        avg_discount=("discount", "mean"),
    ).reset_index().sort_values("avg_discount")

    return summary


def discount_correlation(df: pd.DataFrame) -> float:
    """Korelasi antara discount dan profit margin (negatif = makin diskon, makin tergerus profit)."""
    return df["discount"].corr(df["profit_margin"])


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data("../data/Sample_-_Superstore.csv")

    print("=== RFM Sample ===")
    rfm = compute_rfm(df)
    print(rfm.head())

    print("\n=== K-Means Segmentation ===")
    rfm_clustered = kmeans_segmentation(rfm)
    print(rfm_clustered["cluster_label"].value_counts())

    print("\n=== Forecast 3 bulan ke depan ===")
    print(forecast_sales(df, 3).tail(5))

    print("\n=== Top 5 Produk ===")
    print(top_products(df, "sales", 5)[["product_name", "total_sales"]])

    print("\n=== Frequently Bought Together ===")
    print(frequently_bought_together(df, min_orders=15, top_n=5))