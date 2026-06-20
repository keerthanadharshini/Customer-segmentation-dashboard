import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Customer Segmentation Dashboard")
st.markdown(
    "Segment customers by **Annual Income** and **Spending Score** using "
    "K-Means clustering, and explore each segment's profile and business value."
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


raw_df = load_data("customers.csv")


@st.cache_data
def run_kmeans(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Cluster customers on Annual Income & Spending Score and label
    each cluster with a human-readable, business-friendly name."""

    features = df[["Annual Income", "Spending Score"]].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(scaled_features)

    result = df.copy()
    result["Cluster"] = cluster_ids

    income_median = df["Annual Income"].median()
    spending_median = df["Spending Score"].median()

    label_map = {
        ("High", "High"): "Premium Customers",
        ("High", "Low"): "Conservative Savers",
        ("Low", "High"): "Budget Shoppers",
        ("Low", "Low"): "Low-Engagement Customers",
    }

    cluster_labels = {}
    for cluster_id in sorted(result["Cluster"].unique()):
        cluster_rows = result[result["Cluster"] == cluster_id]
        income_level = "High" if cluster_rows["Annual Income"].mean() >= income_median else "Low"
        spending_level = "High" if cluster_rows["Spending Score"].mean() >= spending_median else "Low"
        base_label = label_map[(income_level, spending_level)]

        final_label = base_label
        suffix = 2
        while final_label in cluster_labels.values():
            final_label = f"{base_label} ({suffix})"
            suffix += 1
        cluster_labels[cluster_id] = final_label

    result["Segment"] = result["Cluster"].map(cluster_labels)
    return result



st.sidebar.header("⚙️ Segmentation Settings")

n_clusters = st.sidebar.slider(
    "Number of Segments (K)",
    min_value=2,
    max_value=6,
    value=3,
    help="Number of customer segments for K-Means to create.",
)


clustered_df = run_kmeans(raw_df, n_clusters)

st.sidebar.header("🔎 Filters")


min_age, max_age = int(raw_df["Age"].min()), int(raw_df["Age"].max())
selected_age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
)


all_genders = sorted(raw_df["Gender"].unique())
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=all_genders,
    default=all_genders,
)



mask = (
    clustered_df["Age"].between(selected_age_range[0], selected_age_range[1])
    & clustered_df["Gender"].isin(selected_genders)
)
filtered_df = clustered_df[mask]

if filtered_df.empty:
    st.warning("No customers match the selected filters. Please adjust your selections.")
    st.stop()



total_customers = len(filtered_df)
avg_income = filtered_df["Annual Income"].mean()
avg_spending = filtered_df["Spending Score"].mean()
num_segments = filtered_df["Segment"].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("👥 Total Customers", f"{total_customers:,}")
kpi2.metric("💰 Average Income", f"${avg_income:,.0f}")
kpi3.metric("🛍️ Average Spending Score", f"{avg_spending:.1f} / 100")
kpi4.metric("🧩 Number of Segments", f"{num_segments}")

st.markdown("---")


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.subheader("📍 Customer Segmentation")
    fig_scatter = px.scatter(
        filtered_df,
        x="Annual Income",
        y="Spending Score",
        color="Segment",
        hover_data=["Customer ID", "Age", "Gender"],
        title="Segments by Income & Spending Score",
    )
    fig_scatter.update_traces(marker=dict(size=9, opacity=0.8))
    fig_scatter.update_layout(legend_title_text="Segment")
    st.plotly_chart(fig_scatter, use_container_width=True)


with chart_col2:
    st.subheader("📊 Segment Distribution")
    segment_counts = filtered_df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Customer Count"]
    fig_bar = px.bar(
        segment_counts,
        x="Segment",
        y="Customer Count",
        color="Segment",
        title="Customers per Segment",
        text="Customer Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")


st.subheader("📋 Segment Summary")

segment_summary = (
    filtered_df.groupby("Segment")
    .agg(
        Customer_Count=("Customer ID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual Income", "mean"),
        Avg_Spending_Score=("Spending Score", "mean"),
    )
    .reset_index()
    .sort_values("Avg_Income", ascending=False)
)

segment_summary = segment_summary.rename(
    columns={
        "Customer_Count": "Customer Count",
        "Avg_Age": "Avg. Age",
        "Avg_Income": "Avg. Income ($)",
        "Avg_Spending_Score": "Avg. Spending Score",
    }
)
segment_summary["Avg. Age"] = segment_summary["Avg. Age"].round(1)
segment_summary["Avg. Income ($)"] = segment_summary["Avg. Income ($)"].round(0)
segment_summary["Avg. Spending Score"] = segment_summary["Avg. Spending Score"].round(1)

st.dataframe(segment_summary, use_container_width=True, hide_index=True)



st.subheader("💡 Business Insights by Segment")


insight_library = {
    "Premium Customers": (
        "High income and high spending. These are the most valuable "
        "customers and the primary revenue drivers.",
        "Prioritize retention with loyalty programs, early access to new "
        "products, and personalized premium offers.",
    ),
    "Conservative Savers": (
        "High income but low spending. They have strong purchasing power "
        "that isn't being converted into sales.",
        "Use targeted promotions, limited-time discounts, or curated "
        "recommendations to nudge them toward higher spending.",
    ),
    "Budget Shoppers": (
        "Lower income but high spending score, suggesting frequent or "
        "impulsive purchases relative to their budget.",
        "Offer value bundles, installment/financing options, and loyalty "
        "rewards to retain volume while protecting margins.",
    ),
    "Low-Engagement Customers": (
        "Lower income and low spending. This segment contributes the least "
        "revenue and shows limited engagement.",
        "Re-engage with low-cost entry offers, awareness campaigns, or "
        "consider lower-priority marketing spend on this group.",
    ),
}


for _, row in segment_summary.iterrows():
    segment_name = row["Segment"]
    base_name = segment_name.split(" (")[0]
    description, recommendation = insight_library.get(
        base_name,
        ("Profile not classified.", "Review this segment's data manually."),
    )

    with st.expander(f"🔹 {segment_name}  —  {int(row['Customer Count'])} customers"):
        st.markdown(f"**Profile:** {description}")
        st.markdown(f"**Recommended Action:** {recommendation}")
        st.markdown(
            f"**Averages:** Age {row['Avg. Age']} · "
            f"Income ${row['Avg. Income ($)']:,.0f} · "
            f"Spending Score {row['Avg. Spending Score']}"
        )



with st.expander("🔍 View Filtered Customer Data with Segment Labels"):
    st.dataframe(
        filtered_df[["Customer ID", "Age", "Gender", "Annual Income", "Spending Score", "Segment"]],
        use_container_width=True,
        hide_index=True,
    )
