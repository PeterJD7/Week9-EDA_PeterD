import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------------------
# 1. LOAD DATA
# -------------------------------------
@st.cache_data
def load_data():
    ec2 = pd.read_csv("aws_resources_compute.csv")
    s3 = pd.read_csv("aws_resources_S3.csv")
    return ec2, s3

ec2, s3 = load_data()

st.set_page_config(page_title="AWS Resource Dashboard", layout="wide")
st.title("☁️ AWS Resource Analysis Dashboard")
st.markdown("Interactive dashboard analyzing EC2 and S3 costs, usage, and optimization opportunities.")

# -------------------------------------
# 2. FILTERS
# -------------------------------------
st.sidebar.header("🔍 Filters")

region_filter_ec2 = st.sidebar.multiselect(
    "Select EC2 Regions", ec2["Region"].unique(), default=list(ec2["Region"].unique())
)
instance_filter = st.sidebar.multiselect(
    "Select Instance Types", ec2["InstanceType"].unique(), default=list(ec2["InstanceType"].unique())
)

region_filter_s3 = st.sidebar.multiselect(
    "Select S3 Regions", s3["Region"].unique(), default=list(s3["Region"].unique())
)
storage_filter = st.sidebar.multiselect(
    "Select Storage Classes", s3["StorageClass"].unique(), default=list(s3["StorageClass"].unique())
)

ec2_filtered = ec2[(ec2["Region"].isin(region_filter_ec2)) & (ec2["InstanceType"].isin(instance_filter))]
s3_filtered = s3[(s3["Region"].isin(region_filter_s3)) & (s3["StorageClass"].isin(storage_filter))]

st.success(f"Showing {len(ec2_filtered)} EC2 instances and {len(s3_filtered)} S3 buckets after filtering.")

# -------------------------------------
# 3. DATA CLEANING & OUTLIERS
# -------------------------------------
ec2_filtered.fillna({"CPUUtilization": ec2_filtered["CPUUtilization"].mean(),
                     "MemoryUtilization": ec2_filtered["MemoryUtilization"].mean()}, inplace=True)
s3_filtered.fillna({"TotalSizeGB": s3_filtered["TotalSizeGB"].mean()}, inplace=True)

# -------------------------------------
# 4. MAIN VISUALIZATIONS
# -------------------------------------
st.header("📊 General Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.subheader("EC2: CPU Utilization Histogram")
    fig, ax = plt.subplots()
    sns.histplot(ec2_filtered["CPUUtilization"], bins=20, kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

with col2:
    st.subheader("EC2: CPU vs Cost Scatter")
    fig, ax = plt.subplots()
    sns.scatterplot(data=ec2_filtered, x="CPUUtilization", y="CostUSD", hue="Region", ax=ax)
    st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    st.subheader("S3: Total Storage by Region")
    s3_region_storage = s3_filtered.groupby("Region")["TotalSizeGB"].sum().sort_values(ascending=False)
    st.bar_chart(s3_region_storage)

with col4:
    st.subheader("S3: Cost vs Storage Scatter")
    fig, ax = plt.subplots()
    sns.scatterplot(data=s3_filtered, x="TotalSizeGB", y="CostUSD", hue="Region", ax=ax)
    st.pyplot(fig)

# -------------------------------------
# 5. INSIGHTS & METRICS VISUALIZED
# -------------------------------------
st.header("🔍 Insights & Metrics")

col5, col6 = st.columns(2)
with col5:
    st.subheader("Top 5 Most Expensive EC2 Instances")
    top_ec2 = ec2_filtered.nlargest(5, "CostUSD")[["ResourceId", "Region", "CostUSD", "InstanceType"]]
    fig, ax = plt.subplots()
    sns.barplot(data=top_ec2, x="CostUSD", y="ResourceId", hue="Region", ax=ax)
    ax.set_xlabel("Cost (USD)")
    st.pyplot(fig)

with col6:
    st.subheader("Top 5 Largest S3 Buckets")
    top_s3 = s3_filtered.nlargest(5, "TotalSizeGB")[["BucketName", "Region", "TotalSizeGB", "CostUSD"]]
    fig, ax = plt.subplots()
    sns.barplot(data=top_s3, x="TotalSizeGB", y="BucketName", hue="Region", ax=ax)
    ax.set_xlabel("Total Size (GB)")
    st.pyplot(fig)

col7, col8 = st.columns(2)
with col7:
    st.subheader("Average EC2 Cost per Region")
    avg_ec2_cost_region = ec2_filtered.groupby("Region")["CostUSD"].mean().round(2).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_ec2_cost_region, x="Region", y="CostUSD", ax=ax, palette="Blues_d")
    ax.set_ylabel("Avg Cost (USD)")
    st.pyplot(fig)

with col8:
    st.subheader("Total S3 Storage per Region")
    total_s3_storage_region = s3_filtered.groupby("Region")["TotalSizeGB"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=total_s3_storage_region, x="Region", y="TotalSizeGB", ax=ax, palette="Greens_d")
    ax.set_ylabel("Total Storage (GB)")
    st.pyplot(fig)

# -------------------------------------
# 6. DYNAMIC OPTIMIZATION INSIGHTS + VISUALS
# -------------------------------------
st.header("⚙️ Optimization Insights (Dynamic & Visual)")

# =============================
# 🖥️ EC2 OPTIMIZATION
# =============================
st.subheader("🖥️ EC2 Optimization Opportunities")

cost_threshold = ec2_filtered["CostUSD"].quantile(0.75)
underutilized = ec2_filtered[
    (ec2_filtered["CPUUtilization"] < 40) & (ec2_filtered["CostUSD"] > cost_threshold)
]

if not underutilized.empty:
    st.markdown(f"**{len(underutilized)} EC2 instances** appear underutilized but expensive.")
    st.dataframe(underutilized[["ResourceId", "Region", "InstanceType", "CostUSD", "CPUUtilization"]])

    # Visualization: EC2 underutilized
    fig, ax = plt.subplots()
    sns.scatterplot(data=ec2_filtered, x="CPUUtilization", y="CostUSD", hue="Region", ax=ax)
    sns.scatterplot(data=underutilized, x="CPUUtilization", y="CostUSD", color="red", s=100, label="Underutilized")
    ax.axvline(40, color="orange", linestyle="--", label="40% Utilization Threshold")
    ax.set_title("EC2 Cost vs CPU Utilization (Optimization Zone)")
    st.pyplot(fig)

    st.info(
        f"💡 These instances have low CPU (<40%) but are in the top 25% of costs. "
        f"Example: `{underutilized.iloc[0]['ResourceId']}` in `{underutilized.iloc[0]['Region']}` "
        f"({underutilized.iloc[0]['InstanceType']}`) could be right-sized or converted to Spot to cut cost."
    )
else:
    st.success("✅ No underutilized high-cost EC2 instances detected based on current filters.")

# =============================
# 🗄️ S3 OPTIMIZATION
# =============================
st.subheader("🗄️ S3 Optimization Opportunities")

# Large unencrypted buckets
large_insecure = s3_filtered[
    (s3_filtered["TotalSizeGB"] > 1000) & (s3_filtered["Encryption"].isin(["None", ""]))
]

# Old costly buckets
s3_filtered["CreationDate"] = pd.to_datetime(s3_filtered["CreationDate"], errors="coerce")
old_costly = s3_filtered[
    (s3_filtered["CostUSD"] > s3_filtered["CostUSD"].quantile(0.75)) &
    (s3_filtered["CreationDate"] < "2023-01-01")
]

# Combine insights
if not large_insecure.empty or not old_costly.empty:
    if not large_insecure.empty:
        st.markdown(f"**{len(large_insecure)} large S3 buckets** found without encryption.")
        st.dataframe(large_insecure[["BucketName", "Region", "TotalSizeGB", "CostUSD", "Encryption"]])

        # Visualization: large, unencrypted S3 buckets
        fig, ax = plt.subplots()
        sns.scatterplot(data=s3_filtered, x="TotalSizeGB", y="CostUSD", hue="Region", ax=ax)
        sns.scatterplot(data=large_insecure, x="TotalSizeGB", y="CostUSD",
                        color="red", s=100, label="Unencrypted Large Buckets")
        ax.axvline(1000, color="orange", linestyle="--", label="1TB Threshold")
        ax.set_title("S3 Cost vs Total Size (Optimization Zone)")
        st.pyplot(fig)

        st.info(
            f"🔐 Buckets such as `{large_insecure.iloc[0]['BucketName']}` "
            f"({large_insecure.iloc[0]['TotalSizeGB']} GB) lack encryption and exceed 1 TB. "
            f"Enable encryption and move cold data to Glacier or IA."
        )

    if not old_costly.empty:
        st.markdown(f"**{len(old_costly)} legacy buckets** are costly and created before 2023.")
        st.dataframe(old_costly[["BucketName", "Region", "CreationDate", "CostUSD", "TotalSizeGB"]])

        # Visualization: old costly buckets
        fig, ax = plt.subplots()
        sns.scatterplot(data=s3_filtered, x="CreationDate", y="CostUSD", hue="Region", ax=ax)
        sns.scatterplot(data=old_costly, x="CreationDate", y="CostUSD",
                        color="red", s=100, label="Old Costly Buckets")
        ax.set_title("S3 Cost vs Creation Date (Legacy High-Cost Buckets)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.info(
            f"🕓 Old buckets like `{old_costly.iloc[0]['BucketName']}` (created {old_costly.iloc[0]['CreationDate'].date()}) "
            f"show high costs. Consider **archiving** older objects or applying **lifecycle rules**."
        )
else:
    st.success("✅ All S3 buckets appear properly encrypted, current, and within cost norms.")


st.header("🤖 EC2 Cost Prediction Model")

# Select numeric features
features = ["CPUUtilization", "MemoryUtilization", "NetworkIn_Bps", "NetworkOut_Bps"]
df_model = ec2_filtered.dropna(subset=features + ["CostUSD"])

if len(df_model) > 5:
    X = df_model[features]
    y = df_model["CostUSD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("Model R²", f"{r2:.2f}")
    st.metric("Mean Absolute Error", f"${mae:.2f}")

    # Interactive prediction
    st.subheader("🔮 Predict EC2 Cost from Utilization")
    cpu = st.slider("CPU Utilization (%)", 0, 100, 60)
    mem = st.slider("Memory Utilization (%)", 0, 100, 60)
    net_in = st.number_input("Network In (Bps)", 0, 10_000_000, 500_000)
    net_out = st.number_input("Network Out (Bps)", 0, 10_000_000, 250_000)

    predicted_cost = model.predict([[cpu, mem, net_in, net_out]])[0]
    st.success(f"💰 Predicted Monthly Cost: **${predicted_cost:.2f}**")

    # Visualization of model fit
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="blue", alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Cost")
    ax.set_ylabel("Predicted Cost")
    ax.set_title("Model Fit: Actual vs Predicted EC2 Cost")
    st.pyplot(fig)

else:
    st.warning("Not enough EC2 data for model training.")


st.header("📦 S3 Bucket Pattern Clustering")

features = ["CostUSD", "TotalSizeGB", "ObjectCount"]
s3_model = s3_filtered.dropna(subset=features)

if len(s3_model) > 3:
    X = s3_model[features]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    s3_model["Cluster"] = kmeans.fit_predict(X_scaled)

    st.dataframe(s3_model[["BucketName", "Region", "CostUSD", "TotalSizeGB", "Cluster"]])

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=s3_model, x="TotalSizeGB", y="CostUSD", hue="Cluster", palette="Set2", ax=ax
    )
    ax.set_title("S3 Cost vs Size (Clustered Patterns)")
    st.pyplot(fig)

    high_cost_cluster = s3_model.groupby("Cluster")["CostUSD"].mean().idxmax()
    st.info(f"💡 Buckets in Cluster {high_cost_cluster} have the highest average cost — consider deeper analysis.")
else:
    st.warning("Not enough S3 data to perform clustering.")


# ============================================================
# 7. MACHINE-LEARNING–ENHANCED OPTIMIZATION RECOMMENDATIONS
# ============================================================
st.header("🧠 Intelligent ML-Based Recommendations")

# ------------------------------------------------------------
# 🔹 EC2: Predictive Cost Analysis (Regression)
# ------------------------------------------------------------
st.subheader("💰 EC2 Predictive Cost Analysis")

features = ["CPUUtilization", "MemoryUtilization", "NetworkIn_Bps", "NetworkOut_Bps"]
df_ec2_model = ec2_filtered.dropna(subset=features + ["CostUSD"])

if len(df_ec2_model) > 10:
    X = df_ec2_model[features]
    y = df_ec2_model["CostUSD"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    df_ec2_model["PredictedCost"] = reg.predict(X)

    # Calculate cost deviation ratio
    df_ec2_model["Deviation"] = (df_ec2_model["PredictedCost"] - df_ec2_model["CostUSD"]) / df_ec2_model["CostUSD"] * 100
    high_risk = df_ec2_model[df_ec2_model["Deviation"] > 15]

    st.metric("Model Mean Absolute Error", f"${mae:.2f}")
    st.metric("At-Risk EC2 Instances", len(high_risk))

    # Visualization of predicted vs actual cost
    fig, ax = plt.subplots()
    ax.scatter(df_ec2_model["CostUSD"], df_ec2_model["PredictedCost"], alpha=0.7)
    ax.plot([df_ec2_model["CostUSD"].min(), df_ec2_model["CostUSD"].max()],
            [df_ec2_model["CostUSD"].min(), df_ec2_model["CostUSD"].max()],
            "r--", label="Perfect Fit")
    ax.set_xlabel("Actual Cost ($)")
    ax.set_ylabel("Predicted Cost ($)")
    ax.set_title("EC2 Predicted vs Actual Cost")
    ax.legend()
    st.pyplot(fig)

    # Highlight and recommend actions
    if not high_risk.empty:
        st.warning("⚠️ EC2 instances likely to exceed expected cost:")
        st.dataframe(high_risk[["ResourceId", "Region", "InstanceType", "CostUSD", "PredictedCost", "Deviation"]])

        sample = high_risk.iloc[0]
        st.info(
            f"💡 **Recommendation:** Instance `{sample['ResourceId']}` in `{sample['Region']}` "
            f"({sample['InstanceType']}`) shows a **{sample['Deviation']:.1f}% higher predicted cost**. "
            f"Investigate high utilization or consider migrating to a smaller instance type or Savings Plan."
        )
    else:
        st.success("✅ No EC2 instances predicted to exceed cost expectations.")
else:
    st.warning("Not enough EC2 data to train predictive model.")


# ------------------------------------------------------------
# 🔹 S3: Pattern Detection via Clustering
# ------------------------------------------------------------
st.subheader("📦 S3 Bucket Pattern Clustering")

features = ["CostUSD", "TotalSizeGB", "ObjectCount"]
df_s3_model = s3_filtered.dropna(subset=features)

if len(df_s3_model) > 5:
    X = df_s3_model[features]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_s3_model["Cluster"] = kmeans.fit_predict(X_scaled)

    # Cluster insights
    cluster_means = df_s3_model.groupby("Cluster")[["CostUSD", "TotalSizeGB"]].mean()
    high_cost_cluster = cluster_means["CostUSD"].idxmax()
    costly_buckets = df_s3_model[df_s3_model["Cluster"] == high_cost_cluster]

    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_s3_model, x="TotalSizeGB", y="CostUSD", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("S3 Bucket Cost Clustering")
    st.pyplot(fig)

    st.metric("High-Cost Cluster ID", int(high_cost_cluster))
    st.metric("Buckets in High-Cost Cluster", len(costly_buckets))

    if not costly_buckets.empty:
        st.warning("⚠️ High-cost S3 buckets detected:")
        st.dataframe(costly_buckets[["BucketName", "Region", "CostUSD", "TotalSizeGB", "ObjectCount", "Cluster"]])

        sample_bucket = costly_buckets.iloc[0]
        st.info(
            f"💡 **Recommendation:** Bucket `{sample_bucket['BucketName']}` in `{sample_bucket['Region']}` "
            f"is in a high-cost cluster with average size {sample_bucket['TotalSizeGB']:.1f} GB. "
            f"Consider lifecycle rules, storage class transitions (e.g., Glacier), "
            f"or deleting stale objects to reduce cost."
        )
    else:
        st.success("✅ No high-cost S3 bucket clusters detected.")
else:
    st.warning("Not enough S3 data for clustering model.")
