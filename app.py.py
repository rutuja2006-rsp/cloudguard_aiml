

import io
import re
import time
import os
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CloudBudget Guard",
    page_icon="☁️",
    layout="wide"
)


# =====================================================
# DARK NAVY THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c32);
    color: #e5edf8;
}

[data-testid="stSidebar"] {
    background: #101a2d;
}

[data-testid="stSidebar"] * {
    color: #e5edf8;
}

h1, h2, h3, h4 {
    color: #f1f5ff !important;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

[data-testid="stMetric"] {
    background: #18263d;
    border: 1px solid #2b3c58;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
}

[data-testid="stFileUploader"] {
    background: #18263d;
    border: 1px dashed #526987;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2b3c58;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER AND CSV UPLOAD
# =====================================================

st.title("☁️ CloudBudget Guard")

st.markdown(
    "Cloud Cost Monitoring | Cost Forecasting | Anomaly Detection"
)

st.divider()

st.subheader("📂 Upload Your Cloud Billing Dataset")

st.write(
    "Upload your CSV dataset to access the dashboard."
)

uploaded_file = st.file_uploader(
    "Choose your cloud cost CSV file",
    type=["csv"],
    key="main_dataset_upload"
)

if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")

    st.markdown("""
    **Required columns**
    - `charge_date` — billing date
    - `cost` — cloud cost amount

    **Optional columns**
    - `provider`
    - `service_name`
    - `team_id`
    - `resource_id`
    - `currency`
    - `budget_amount`
    - `cpu_utilization_pct`
    - `memory_utilization_pct`
    - `network_inbound_bytes`
    - `network_outbound_bytes`
    - `usage_quantity`
    """)

    st.stop()


# =====================================================
# READ AND NORMALIZE DATASET
# =====================================================

try:
    df = pd.read_csv(
        io.BytesIO(uploaded_file.getvalue()),
        low_memory=False
    )

    # Normalize column names
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "date": "charge_date",
        "usage_date": "charge_date",
        "billing_date": "charge_date",
        "transaction_date": "charge_date",
        "service": "service_name",
        "cloud_provider": "provider",
        "team": "team_id",
        "client": "client_id",
        "client_name": "client_id",
        "customer": "client_id",
        "customer_name": "client_id",
        "account_name": "client_id",
        "account_id": "client_id",
        "tenant": "client_id",
        "tenant_id": "client_id",
        "amount": "cost",
        "total_cost": "cost",
        "cost_amount": "cost",
        "currency_code": "currency",
        "cpu_utilization": "cpu_utilization_pct",
        "memory_utilization": "memory_utilization_pct"
    }

    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    required_columns = ["charge_date", "cost"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {missing_columns}. "
            "Your CSV must contain a date column and a cost column."
        )
        st.stop()

    df["charge_date"] = pd.to_datetime(
        df["charge_date"],
        errors="coerce"
    )

    df["cost"] = pd.to_numeric(
        df["cost"],
        errors="coerce"
    )

    df.dropna(
        subset=["charge_date", "cost"],
        inplace=True
    )

    if df.empty:
        st.error(
            "No valid records found. Please check your date and cost values."
        )
        st.stop()

    # Optional columns
    optional_defaults = {
        "provider": "Unknown",
        "service_name": "Unknown",
        "team_id": "Unknown",
        "client_id": "Unknown",
        "resource_id": "Unknown",
        "currency": "USD"
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

        df[col] = df[col].fillna(default).astype(str)

    # Numeric columns
    numeric_columns = [
        "budget_amount",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "network_inbound_bytes",
        "network_outbound_bytes",
        "usage_quantity"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df["month"] = (
        df["charge_date"].dt.to_period("M").astype(str)
    )

except Exception as e:
    st.error(f"Unable to process the uploaded CSV: {e}")
    st.stop()


# =====================================================
# CURRENCY
# =====================================================

currency_values = df["currency"].dropna().unique().tolist()

if len(currency_values) == 1:
    currency = currency_values[0]
else:
    currency = st.selectbox(
        "Select currency for display",
        currency_values
    )

symbol_map = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£"
}

symbol = symbol_map.get(currency.upper(), f"{currency} ")

st.success(
    f"Dataset loaded successfully — {len(df):,} valid records"
)

with st.expander("Preview Uploaded Dataset"):
    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("☁️ CloudBudget Guard")
st.sidebar.success("Dataset loaded")

page = st.sidebar.radio(
    "Navigation",
    [
        "Cloud Cost Dashboard",
        "Cost Forecasting",
        "Anomaly Detection"
    ]
)


# =====================================================
# GLOBAL DATE FILTER
# =====================================================

min_date = df["charge_date"].min().date()
max_date = df["charge_date"].max().date()

st.sidebar.subheader("Global Date Filter")

start_date = st.sidebar.date_input(
    "Start Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
    key="global_start"
)

end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    key="global_end"
)

if start_date > end_date:
    st.error("Start date cannot be after end date.")
    st.stop()

base = df[
    (df["charge_date"].dt.date >= start_date)
    & (df["charge_date"].dt.date <= end_date)
].copy()

# =====================================================
# GLOBAL CLOUD PROVIDER, CLIENT AND SERVICE FILTERS
# These filters apply to all three features.
# =====================================================

st.sidebar.subheader("Cloud Filters")

provider_options = sorted(base["provider"].dropna().astype(str).unique().tolist())
selected_providers = st.sidebar.multiselect(
    "Cloud Provider",
    options=provider_options,
    default=provider_options,
    key="global_provider_filter"
)

provider_filtered = base[base["provider"].astype(str).isin(selected_providers)].copy()

client_options = sorted(provider_filtered["client_id"].dropna().astype(str).unique().tolist())
selected_clients = st.sidebar.multiselect(
    "Client",
    options=client_options,
    default=client_options,
    key="global_client_filter"
)

client_filtered = provider_filtered[provider_filtered["client_id"].astype(str).isin(selected_clients)].copy()

service_options = sorted(client_filtered["service_name"].dropna().astype(str).unique().tolist())
selected_services = st.sidebar.multiselect(
    "Service",
    options=service_options,
    default=service_options,
    key="global_service_filter"
)

base = client_filtered[client_filtered["service_name"].astype(str).isin(selected_services)].copy()

st.sidebar.caption(
    f"Showing {len(base):,} of {len(df):,} records after all filters."
)

if base.empty:
    st.warning("No records match the selected date, provider, client and service filters. Please change your filters.")
    st.stop()
# =====================================================
# 1. OVERVIEW
# =====================================================

if page == "Cloud Cost Dashboard":

    st.header("📊 Cloud Cost Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Cloud Cost",
        f"{symbol}{base['cost'].sum():,.2f}"
    )

    c2.metric(
        "Average Record Cost",
        f"{symbol}{base['cost'].mean():,.2f}"
    )

    c3.metric(
        "Total Records",
        f"{len(base):,}"
    )

    c4.metric(
        "Number of Services",
        f"{base['service_name'].nunique():,}"
    )

    monthly = (
        base.groupby("month", as_index=False)["cost"]
        .sum()
        .sort_values("month")
    )

    st.subheader("Monthly Cloud Spending")

    st.plotly_chart(
        px.line(
            monthly,
            x="month",
            y="cost",
            markers=True,
            title="Monthly Cloud Cost"
        ),
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        service_cost = (
            base.groupby("service_name", as_index=False)["cost"]
            .sum()
            .sort_values("cost", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                service_cost,
                x="service_name",
                y="cost",
                title="Cost by Service",
                color="cost"
            ),
            use_container_width=True
        )

    with right:
        provider_cost = (
            base.groupby("provider", as_index=False)["cost"]
            .sum()
        )

        st.plotly_chart(
            px.pie(
                provider_cost,
                names="provider",
                values="cost",
                title="Cost by Cloud Provider"
            ),
            use_container_width=True
        )

    team_cost = (
        base.groupby("team_id", as_index=False)["cost"]
        .sum()
        .sort_values("cost", ascending=False)
    )

    st.plotly_chart(
        px.bar(
            team_cost,
            x="team_id",
            y="cost",
            title="Cost by Team"
        ),
        use_container_width=True
    )


# =====================================================
# 2. COST FORECASTING
# =====================================================

elif page == "Cost Forecasting":

    st.header("📈 Cost Forecasting")

    monthly = (
        base.groupby("month", as_index=False)["cost"]
        .sum()
        .sort_values("month")
    )

    if len(monthly) < 3:
        st.warning(
            "At least 3 months of data are needed for forecasting."
        )

    else:
        horizon = st.selectbox(
            "Forecast Horizon",
            [1, 2, 3, 6, 12],
            index=2,
            format_func=lambda x: f"{x} month(s)"
        )

        monthly["month_date"] = pd.to_datetime(
            monthly["month"] + "-01"
        )

        monthly["index"] = np.arange(len(monthly))

        model = LinearRegression()

        model.fit(
            monthly[["index"]],
            monthly["cost"]
        )

        future_index = np.arange(
            len(monthly),
            len(monthly) + horizon
        ).reshape(-1, 1)

        future_dates = pd.date_range(
            monthly["month_date"].iloc[-1]
            + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS"
        )

        predictions = np.maximum(
            model.predict(future_index),
            0
        )

        future = pd.DataFrame({
            "month": future_dates.strftime("%Y-%m"),
            "cost": predictions,
            "type": "Forecast"
        })

        history = monthly[["month", "cost"]].copy()
        history["type"] = "Historical"

        chart_data = pd.concat(
            [history, future],
            ignore_index=True
        )

        st.metric(
            "Next Month Forecast",
            f"{symbol}{predictions[0]:,.2f}"
        )

        st.plotly_chart(
            px.line(
                chart_data,
                x="month",
                y="cost",
                color="type",
                markers=True,
                title="Historical Cost and Forecast"
            ),
            use_container_width=True
        )

        st.dataframe(
            future,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Forecast CSV",
            future.to_csv(index=False).encode("utf-8"),
            "cost_forecast.csv",
            "text/csv"
        )


# =====================================================
# 3. ANOMALY DETECTION
# =====================================================

elif page == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")
    st.caption(f"Analysis uses the shared date range: {start_date} to {end_date}.")

    selected = base.copy()

    if len(selected) < 10:
        st.warning("The selected date range must contain at least 10 records.")
    elif selected["cost"].nunique() < 2:
        st.warning("Cost values must vary to identify unusual records.")
    else:
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42
        )
        selected["prediction"] = model.fit_predict(selected[["cost"]])
        selected["anomaly_score"] = model.decision_function(selected[["cost"]])
        anomalies = selected[selected["prediction"] == -1].copy()
        anomalies = anomalies.sort_values("anomaly_score")

        a, b, c = st.columns(3)
        a.metric("Records Analysed", f"{len(selected):,}")
        b.metric("Potential Anomalies", f"{len(anomalies):,}")
        c.metric("Anomaly Share", f"{len(anomalies) / len(selected) * 100:.2f}%")

        st.subheader("Potentially Unusual Records")
        show_cols = [
            col for col in [
                "charge_date", "provider", "client_id", "service_name", "team_id",
                "resource_id", "cost", "currency", "anomaly_score"
            ] if col in anomalies.columns
        ]

        if anomalies.empty:
            st.success("No unusual records were detected.")
        else:
            st.dataframe(anomalies[show_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Anomaly Results",
                anomalies[show_cols].to_csv(index=False).encode("utf-8"),
                "anomaly_results.csv",
                "text/csv"
            )
# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    "**CloudBudget Guard** · Cloud Spend Analytics Dashboard"
)

