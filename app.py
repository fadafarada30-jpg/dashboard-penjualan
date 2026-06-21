"""
app.py
Dashboard Interaktif Analisis Penjualan - Superstore
Jalankan dengan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data, get_kpi_summary
from utils.analysis import (
    compute_rfm,
    kmeans_segmentation,
    forecast_sales,
    top_products,
    frequently_bought_together,
    sales_by_state,
    discount_impact_analysis,
    discount_correlation,
)

# ---------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Penjualan",
    page_icon="📊",
    layout="wide",
)

st.title("Dashboard Analisis Penjualan & Performa Seller")
st.caption("Data: Superstore Sales Dataset | Dibuat untuk Projek Akhir Algoritma Data Science")

# ---------------------------------------------------------
# CUSTOM CSS - styling tambahan di atas theme bawaan
# ---------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1A1F2C;
        border: 1px solid #2E86AB;
        border-radius: 10px;
        padding: 15px 10px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 500;
    }
    h1 {
        background: linear-gradient(90deg, #2E86AB, #5DD3C4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def get_data():
    return load_data("data/Sample_-_Superstore.csv")

df = get_data()

# ---------------------------------------------------------
# SIDEBAR - FILTER INTERAKTIF
# ---------------------------------------------------------
st.sidebar.header("Filter Data")

min_date, max_date = df["order_date"].min(), df["order_date"].max()
date_range = st.sidebar.date_input(
    "Rentang Tanggal Order",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

regions = st.sidebar.multiselect(
    "Region", options=sorted(df["region"].unique()), default=sorted(df["region"].unique())
)

categories = st.sidebar.multiselect(
    "Kategori", options=sorted(df["category"].unique()), default=sorted(df["category"].unique())
)

segments = st.sidebar.multiselect(
    "Segmen Customer", options=sorted(df["segment"].unique()), default=sorted(df["segment"].unique())
)

# Apply filter
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df[
        (df["order_date"] >= start_date)
        & (df["order_date"] <= end_date)
        & (df["region"].isin(regions))
        & (df["category"].isin(categories))
        & (df["segment"].isin(segments))
    ]
else:
    df_filtered = df.copy()

if df_filtered.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Coba ubah filter di sidebar.")
    st.stop()

# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
kpi = get_kpi_summary(df_filtered)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"${kpi['total_sales']:,.0f}")
col2.metric("📈 Total Profit", f"${kpi['total_profit']:,.0f}")
col3.metric("🧾 Total Orders", f"{kpi['total_orders']:,}")
col4.metric("📊 Profit Margin", f"{kpi['profit_margin']*100:.1f}%")

st.divider()

# ---------------------------------------------------------
# TABS UNTUK SETIAP JENIS ANALISIS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Tren & Overview", "Produk Terlaris", "Segmentasi Customer",
     "Forecasting", "Sering Dibeli Bareng", "Peta Sebaran", "Dampak Diskon"]
)

# --- TAB 1: TREN & OVERVIEW ---
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        monthly_sales = df_filtered.groupby("order_year_month")["sales"].sum().reset_index()
        fig = px.line(monthly_sales, x="order_year_month", y="sales", markers=True,
                       title="Tren Penjualan Bulanan")
        fig.update_layout(xaxis_title="Bulan", yaxis_title="Total Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        region_sales = df_filtered.groupby("region")["sales"].sum().reset_index()
        fig = px.bar(region_sales, x="region", y="sales", title="Sales per Region", color="region")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        cat_sales = df_filtered.groupby("category")["sales"].sum().reset_index()
        fig = px.pie(cat_sales, names="category", values="sales", title="Komposisi Sales per Kategori")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        sub_profit = df_filtered.groupby("sub_category")["profit"].sum().reset_index().sort_values("profit")
        fig = px.bar(sub_profit, x="profit", y="sub_category", orientation="h",
                      title="Profit per Sub-Kategori", color="profit",
                      color_continuous_scale=["red", "green"])
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PRODUK TERLARIS ---
with tab2:
    metric_choice = st.radio("Urutkan berdasarkan:", ["sales", "profit", "quantity"], horizontal=True)
    n_top = st.slider("Jumlah produk yang ditampilkan", 5, 20, 10)

    top_df = top_products(df_filtered, by=metric_choice, n=n_top)
    col_label = f"total_{metric_choice}"

    fig = px.bar(top_df, x=col_label, y="product_name", orientation="h",
                  title=f"Top {n_top} Produk berdasarkan {metric_choice.capitalize()}",
                  color="category")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(top_df, use_container_width=True)

# --- TAB 3: SEGMENTASI CUSTOMER (RFM + K-MEANS) ---
with tab3:
    st.subheader("Segmentasi Customer dengan RFM Analysis + K-Means Clustering")
    st.caption("RFM = Recency (kebaruan transaksi), Frequency (frekuensi order), Monetary (total belanja)")

    rfm = compute_rfm(df_filtered)
    n_clusters = st.slider("Jumlah Cluster (K-Means)", 2, 6, 4)
    rfm_clustered = kmeans_segmentation(rfm, n_clusters=n_clusters)

    c1, c2 = st.columns(2)
    with c1:
        seg_count = rfm_clustered["cluster_label"].value_counts().reset_index()
        seg_count.columns = ["cluster_label", "count"]
        fig = px.pie(seg_count, names="cluster_label", values="count", title="Distribusi Cluster Customer")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            rfm_clustered, x="frequency", y="monetary", color="cluster_label",
            size="recency", hover_data=["customer_name"],
            title="Scatter Plot: Frequency vs Monetary (ukuran bubble = Recency)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Detail per Customer**")
    st.dataframe(
        rfm_clustered[["customer_name", "recency", "frequency", "monetary", "segment_label", "cluster_label"]]
        .sort_values("monetary", ascending=False),
        use_container_width=True,
    )

# --- TAB 4: FORECASTING ---
with tab4:
    st.subheader("Prediksi Penjualan Bulan ke Depan")
    periods = st.slider("Berapa bulan ke depan diprediksi?", 1, 6, 3)

    forecast_df = forecast_sales(df_filtered, periods_ahead=periods)
    model_used = forecast_df.attrs.get("model_used", "Linear Regression")

    if "Holt-Winters" in model_used:
        st.success(f"✅ Model yang dipakai: **{model_used}** — menangkap tren & pola musiman tahunan.")
    else:
        st.warning(f"⚠️ Model yang dipakai: **{model_used}**. Holt-Winters butuh data ≥24 bulan "
                   "dan library `statsmodels` ter-install. Coba kurangi filter tanggal atau install "
                   "statsmodels untuk hasil lebih akurat.")

    fig = px.line(forecast_df, x="order_year_month", y="sales", color="type", markers=True,
                  title="Actual vs Forecast Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df.tail(periods + 3), use_container_width=True)

# --- TAB 5: MARKET BASKET (PRODUK SERING DIBELI BARENG) ---
with tab5:
    st.subheader("Sub-Kategori Produk yang Sering Dibeli Bersamaan")
    min_co = st.slider("Minimum co-occurrence", 2, 50, 10)

    pairs = frequently_bought_together(df_filtered, min_orders=min_co, top_n=15)

    if pairs.empty:
        st.warning("Tidak ada pasangan produk yang memenuhi threshold ini. Coba turunkan nilai minimum.")
    else:
        fig = px.bar(
            pairs, x="co_occurrence", y=pairs["product_a"] + " + " + pairs["product_b"],
            orientation="h", title="Pasangan Sub-Kategori Paling Sering Muncul Bersama"
        )
        fig.update_layout(yaxis_title="Pasangan Produk")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pairs, use_container_width=True)

# --- TAB 6: PETA SEBARAN (GEOGRAPHIC ANALYSIS) ---
with tab6:
    st.subheader("Sebaran Sales & Profit per State")

    state_df = sales_by_state(df_filtered)
    map_metric = st.radio("Tampilkan berdasarkan:", ["total_sales", "profit_margin"], horizontal=True,
                           format_func=lambda x: "Total Sales" if x == "total_sales" else "Profit Margin")

    fig = px.choropleth(
        state_df, locations="state_code", locationmode="USA-states", color=map_metric,
        scope="usa", color_continuous_scale="RdYlGn" if map_metric == "profit_margin" else "Blues",
        hover_name="state", hover_data=["total_sales", "total_profit", "total_orders"],
        title=f"Peta {'Total Sales' if map_metric == 'total_sales' else 'Profit Margin'} per State"
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🏆 Top 5 State - Sales Tertinggi**")
        st.dataframe(state_df.nlargest(5, "total_sales")[["state", "total_sales", "profit_margin"]],
                     use_container_width=True)
    with c2:
        st.markdown("**⚠️ Bottom 5 State - Profit Margin Terendah**")
        st.dataframe(state_df.nsmallest(5, "profit_margin")[["state", "total_sales", "profit_margin"]],
                     use_container_width=True)

# --- TAB 7: DAMPAK DISKON KE PROFIT ---
with tab7:
    st.subheader("Analisis Dampak Diskon terhadap Profit")
    st.caption("Membantu seller menentukan level diskon yang masih 'aman' sebelum profit tergerus signifikan.")

    corr = discount_correlation(df_filtered)
    corr_strength = "sangat kuat" if abs(corr) > 0.7 else "cukup kuat" if abs(corr) > 0.4 else "lemah"
    direction = "semakin besar diskon, semakin rendah profit margin" if corr < 0 else "diskon tidak terlalu menggerus profit"

    st.metric("Korelasi Diskon vs Profit Margin", f"{corr:.3f}")
    st.info(f"💡 Korelasinya **{corr_strength}** ({direction}).")

    impact_df = discount_impact_analysis(df_filtered)

    fig = px.bar(
        impact_df, x="discount_bucket_label", y="avg_profit_margin",
        title="Rata-rata Profit Margin per Rentang Diskon",
        color="avg_profit_margin", color_continuous_scale="RdYlGn",
        labels={"discount_bucket_label": "Rentang Diskon", "avg_profit_margin": "Avg Profit Margin"}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(impact_df, use_container_width=True)

st.divider()
st.caption("Dibuat dengan Streamlit, Pandas, Scikit-learn & Plotly | Projek Akhir Algoritma Data Science")