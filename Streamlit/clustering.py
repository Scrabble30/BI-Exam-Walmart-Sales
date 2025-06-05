import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def clustering():
    # ============================================================
    # RQ2: Clustering Store Sales Patterns
    # ============================================================
    st.markdown("""
    <h1 style="margin-bottom:0.3em">RQ2: Clustering Store Sales Patterns</h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    **Objective:** Identify natural groupings of stores based on their sales patterns and local economic context to inform business strategy.
    - **Research Question (RQ2):** What natural groupings of store sales patterns can be identified?
    - **Hypothesis (H):** Clustering stores on average weekly sales, fuel price, CPI, and unemployment will uncover distinct profiles (e.g., “holiday-sensitive”, “price-sensitive”).

    This analysis is structured step-by-step with detailed explanations, numeric outputs, and visualizations—designed for readers new to business intelligence. You'll see **what** we do, **how** we do it, and **why** each step matters.
    """)

    st.markdown("""
    This analysis explores natural groupings (clusters) among Walmart stores based on their sales and economic patterns. The goal is to uncover meaningful store segments to support business strategy, targeted marketing, or inventory planning.
    """)

    # ------------------------ TABLE OF CONTENTS ------------------------
    st.markdown("""
    <details open>
    <summary><h4 style="display:inline; color:#0366d6;">📑 Table of Contents</h4></summary>

    1. [Data Loading & Preview](#data-loading--preview)
    2. [Data Description & Aggregation Rationale](#data-description--aggregation-rationale)
    3. [Initial Summary Statistics](#initial-summary-statistics)
    4. [Outlier Detection & Removal](#outlier-detection--removal)
    5. [Scaling & Transformation](#scaling--transformation)
    6. [Choosing the Number of Clusters (k)](#choosing-the-number-of-clusters-k)
    7. [K-Means Clustering & Cluster Profiles](#k-means-clustering--cluster-profiles)
    8. [Cluster Stability: Assessing Cluster Size Variability](#cluster-stability-assessing-cluster-size-variability)
    9. [PCA for Visualization & Insight](#pca-for-visualization--insight)
    10. [Numeric Cluster Summary & Bar Charts](#numeric-cluster-summary--bar-charts)
    11. [Interpretation & Business Insights](#interpretation--business-insights)
    12. [Expanded Mean-Shift Clustering Implementation](#expanded-mean-shift-clustering-implementation)
    13. [Conclusion](#conclusion)
    </details>
    """, unsafe_allow_html=True)
    st.write("---")

    # ============================================================
    # 1. DATA LOADING & PREVIEW
    # ============================================================
    st.header("1) Data Loading & Preview", anchor="data-loading--preview")
    st.markdown("""
    The first step in any data analysis project is to load the dataset and **gain an initial understanding of its structure, completeness, and quality**. This helps you quickly spot issues (like missing data, duplicates, or outliers) and get a feel for what’s available.
    """)
    with st.expander("Show full checklist for data loading (click to expand)"):
        st.markdown("""
        **Key steps:**  
        - Load the dataset, making sure dates are parsed correctly.
        - Preview the first few rows to understand the columns and their typical values.
        - Examine data types to catch potential issues with parsing (e.g., numbers stored as strings).
        - Check for duplicate or missing weekly records for each store—this ensures data consistency for time-series analysis.
        - Review the time range covered by the data.
        - Assert that key numeric fields (sales, fuel price) are in valid ranges to catch data entry errors early.
        - Summarize the shape and data types of the DataFrame.
        - Assess the presence of missing data to plan for cleaning/imputation steps.

        These checks lay the groundwork for **clean, reliable analysis** and help prevent costly mistakes or misleading insights later on.
        """)
    directory = "../Data/"
    file_name = "Walmart_Sales.csv"
    data_path = directory + file_name
    try:
        df = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()
    st.write(f"Total weekly records: {len(df):,}")
    st.write("Preview of the first 5 rows:")
    st.dataframe(df.head())

    st.write("Data Types:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=['Dtype']))

    dupes = df.groupby('Store')['Date'].nunique().reset_index(name='n_weeks')
    st.write("Number of Unique Weeks per Store (should be equal for all stores):")
    st.dataframe(dupes['n_weeks'].value_counts().to_frame('Stores Count'))

    st.write(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    try:
        assert (df['Weekly_Sales'] >= 0).all(), "Negative sales found!"
        assert (df['Fuel_Price'] > 0).all(), "Invalid (non-positive) fuel prices found!"
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write(f"Rows, columns: {df.shape}")

    missing = df.isnull().sum()
    if (missing > 0).any():
        st.warning("Missing data detected! Consider imputation/cleaning.")
        st.write(missing[missing > 0])
    else:
        st.success("No missing values detected.")
    st.write("---")

    # ============================================================
    # 2. DATA DESCRIPTION & AGGREGATION RATIONALE
    # ============================================================
    st.header("2️) Data Description & Aggregation Rationale", anchor="data-description--aggregation-rationale")
    st.markdown("""
    The raw dataset records **weekly sales at the department level for each store**, alongside economic features such as fuel price, CPI (Consumer Price Index), and unemployment rate.  

    However, clustering on weekly or department-level data would lead to a very large, noisy dataset—and our primary goal is to **identify groups of similar stores** based on their average sales performance and typical economic environment.
    """)
    with st.expander("Show Aggregation Approach and Metrics"):
        st.markdown("""
        **Aggregation Approach:**  
        - **We aggregate all weekly records to a single row per store,** computing the mean for each key variable.
        - This step yields one representative "profile" for each store, smoothing out short-term fluctuations and department-level noise.

        **Metrics after aggregation:**  
        - **Average weekly sales** (`avg_sales`): Captures typical store revenue.
        - **Average fuel price** (`avg_fuel`): Reflects local economic conditions that can impact store traffic and purchasing behavior.
        - **Average CPI** (`avg_CPI`): Indicates cost-of-living or price levels in the area.
        - **Average unemployment** (`avg_unemp`): A proxy for the local labor market and potential consumer demand.

        This aggregated view enables meaningful clustering—each store is now represented by a set of features that describe its "typical" environment and performance, suitable for uncovering natural store groupings.
        """)
    agg = df.groupby('Store').agg(
        avg_sales=('Weekly_Sales', 'mean'),
        avg_fuel=('Fuel_Price', 'mean'),
        avg_CPI=('CPI', 'mean'),
        avg_unemp=('Unemployment', 'mean')
    ).reset_index()

    st.write(f"Total stores after aggregation: {len(agg)}")
    st.dataframe(agg.head())
    st.write("---")

    # ============================================================
    # 3. INITIAL SUMMARY STATISTICS
    # ============================================================
    st.header("3️) Initial Summary Statistics", anchor="initial-summary-statistics")
    st.markdown("""
    Before performing clustering, it's important to **understand the typical values and spread of each key variable** across all stores. This step provides valuable context for interpreting later clustering results and for spotting potential outliers or data quality issues.
    """)
    with st.expander("Show Feature Summary Details"):
        st.markdown("""
        **For each feature, we examine:**
        - **Count:** Number of stores (should match your aggregation result)
        - **Mean & Median:** The average and central tendency for each variable
        - **Standard Deviation:** Indicates the amount of variation or "spread" between stores
        - **Minimum & Maximum:** The range for each metric, helping spot outliers or unusual stores

        Understanding these statistics is crucial for:
        - Selecting features for clustering (avoid highly skewed or redundant variables)
        - Detecting any stores with unusual economic environments or sales patterns
        - Providing a business sense of "typical" store conditions across the network
        """)
    desc = agg[['avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']].describe().transpose()
    pretty_desc = pd.DataFrame(index=desc.index, columns=desc.columns)
    for idx in desc.index:
        for col in desc.columns:
            val = desc.loc[idx, col]
            if idx == 'avg_sales' and col not in ['count']:
                if val >= 1_000_000:
                    pretty_desc.loc[idx, col] = f"${val/1_000_000:.2f}M"
                elif val >= 1_000:
                    pretty_desc.loc[idx, col] = f"${val/1_000:.0f}K"
                else:
                    pretty_desc.loc[idx, col] = f"${val:,.0f}"
            else:
                pretty_desc.loc[idx, col] = f"{round(val, 2)}"
    pretty_desc['count'] = desc['count'].astype(int).astype(str)
    st.dataframe(pretty_desc)
    st.write("---")

    # ============================================================
    # 4. OUTLIER DETECTION & REMOVAL
    # ============================================================
    st.header("4️) Outlier Detection & Removal", anchor="outlier-detection--removal")
    st.markdown("""
    ### Why consider outliers?
    Outliers—stores with extremely high or low average values—can **distort cluster centroids** and create artificial groupings that don’t reflect the majority of stores. Detecting and handling outliers helps ensure that clusters reflect typical store behaviors rather than being skewed by rare, extreme cases.
    """)
    with st.expander("Show Pros and Cons of Outlier Removal"):
        st.markdown("""
        #### **Pros of outlier removal:**
        - Clusters become tighter and more representative of "normal" stores.
        - Analysis focuses on common patterns, not one-off anomalies.

        #### **Cons of outlier removal:**
        - We lose information about unique, possibly strategic stores (e.g., flagship or struggling locations).
        - If your goal is to target or study extremes, don’t drop them!

        #### **Why only remove aggregate-level outliers?**
        - Our clustering operates at the **store level** (one row per store), so only aggregated metrics matter.
        - Weekly spikes or anomalies (such as holidays) are averaged out and don’t overly impact clustering.
        """)
    zs = agg[['avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']].apply(zscore)
    mask_outlier = (zs.abs() >= 3).any(axis=1)
    outliers = agg[mask_outlier]
    clean = agg[~mask_outlier].reset_index(drop=True)
    st.write(f'Outlier stores dropped: {len(outliers)}')
    if not outliers.empty:
        st.dataframe(outliers)
    st.write(f'Stores remaining after removal: {len(clean)}')
    st.dataframe(clean.describe().transpose())

    st.subheader("Correlation Matrix: Exploring Feature Relationships")
    st.markdown("""
    Before clustering, it’s useful to visualize how our features relate to each other:
    - **High positive correlation** means features tend to increase together.
    - **High negative correlation** means features move in opposite directions.
    - **Low correlation** means features provide unique information.

    Strong correlations can indicate redundant features, which may weaken clustering (because the same information is counted multiple times). This step helps you **understand the “shape” of your feature space** and can inform feature selection for better clustering.

    The annotated heatmap makes it easy to spot important relationships (e.g., whether high fuel prices are linked to high or low sales, or if CPI and unemployment move together).
    """)
    corr = clean[['avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax_corr
    )
    ax_corr.set_title('Feature Correlation Matrix (Full)')
    st.pyplot(fig_corr)

    # ============================================================
    # 5. SCALING & TRANSFORMATION
    # ============================================================
    st.write("---")
    st.header("5) Scaling & Transformation", anchor="scaling--transformation")
    st.subheader("6.1 Log Transform of Sales")
    st.markdown("""
    - **Why log-transform sales?**  
    The distribution of store sales is highly **right-skewed**—a few very large stores have much higher sales than the rest. This can dominate clustering (since distance-based methods focus on large absolute differences).
    - **What does log1p do?**  
    Applying the natural log (`np.log1p`) compresses the scale, making the distribution more symmetrical. This gives each store more equal weight, and makes clustering focus on proportional differences rather than just absolute ones.
    - **Visualizations:**  
    - The left histogram shows the original, skewed sales distribution.
    - The right shows the (much more normal) distribution after log transformation.
    - The KDE (density plot) overlays further highlight the reduction in skewness.

    **Takeaway:**  
    *Log-transforming sales is a standard best practice when clustering retail or financial data, ensuring that high-volume stores do not overwhelm the algorithm.*
    """)
    clean['log_sales'] = np.log1p(clean['avg_sales'])
    stats = pd.DataFrame({
        'orig_mean': [clean['avg_sales'].mean()],
        'orig_std':  [clean['avg_sales'].std()],
        'log_mean':  [clean['log_sales'].mean()],
        'log_std':   [clean['log_sales'].std()]
    }, index=['value'])
    st.dataframe(stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(clean['avg_sales'], bins=15)
    axes[0].set_title('Original Sales')
    axes[1].hist(clean['log_sales'], bins=15)
    axes[1].set_title('Log-Transformed Sales')
    st.pyplot(fig)

    st.write("""Above are histograms of store average weekly sales **before** and **after** applying a log transformation.

- **Original Sales (left):** The distribution is heavily right-skewed, with most stores earning below $1.5M, but a few outliers with much higher sales.
- **Log-Transformed Sales (right):** The data is now much more symmetric and "normal," with compressed extremes and a more even spread.

**Why log-transform sales?**  
- Log transformation reduces the impact of extreme outliers and very large stores, ensuring they don’t dominate the clustering algorithm.
- This creates a fairer basis for segmenting stores by their typical sales level rather than their size alone.""")

    fig_kde, ax_kde = plt.subplots(figsize=(10, 4))
    sns.kdeplot(clean['avg_sales'], label='Original', fill=True, ax=ax_kde)
    sns.kdeplot(clean['log_sales'], label='Log-Transformed', fill=True, ax=ax_kde)
    ax_kde.set_title('Sales Distribution: Before vs After Log Transform')
    ax_kde.set_xlabel('Sales')
    ax_kde.legend()
    st.pyplot(fig_kde)

    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=clean, x='avg_unemp', y='avg_sales', s=100, alpha=0.8, ax=ax_scatter)
    ax_scatter.set_title('Strategic Priority: Sales vs Unemployment')
    ax_scatter.set_xlabel('Unemployment Rate (%)')
    ax_scatter.set_ylabel('Avg Sales ($)')
    ax_scatter.axvline(8.5, ls='--', c='gray')
    ax_scatter.text(9, clean['avg_sales'].max()*0.7, 'High Risk: Low Sales + High Unemployment', 
            rotation=90, va='center', fontsize=10, color='red')
    st.pyplot(fig_scatter)
    st.write("""
    ## Strategic Priority: Sales vs. Unemployment

This scatter plot visualizes each store’s average sales against its average local unemployment rate.

- **X-axis:** Unemployment rate (%)
- **Y-axis:** Average weekly sales ($)
- The vertical dashed line indicates a threshold for “high” unemployment (e.g., 8.5%).

**Interpretation:**  
- Stores in the lower right (low sales, high unemployment) are potentially vulnerable and may require additional business support or targeted marketing efforts.
- High-performing stores (top of the plot) in low-unemployment areas represent stability and opportunity for piloting new initiatives.
""")             

    st.markdown("""
    ### 6.2 Standardization

    - **Why standardize?**  
    Clustering is sensitive to variable scales. Features like sales (millions) would drown out others like unemployment (single digits) unless we put them on the same scale.
    - **How?**  
    We use standardization (z-score scaling): mean = 0, std = 1 for each variable. This ensures that **all features contribute equally** to distance calculations.

    **Takeaway:**  
    *Standardization is always needed before clustering when your features are measured in different units or scales.*
    """)
    scaler = StandardScaler()
    features = clean[['log_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']]
    X = scaler.fit_transform(features)
    st.write('Features standardized. All features now have mean 0 and variance 1.')

    # ============================================================
    # 6. CHOOSING THE NUMBER OF CLUSTERS (k)
    # ============================================================
    st.write("---")
    st.header("6️) Choosing the Number of Clusters (k)", anchor="choosing-the-number-of-clusters-k")
    st.markdown("""
    Selecting the optimal number of clusters is critical for effective segmentation. We use two main techniques:
    """)
    with st.expander("Show Full Elbow & Silhouette Explanation"):
        st.markdown("""
        ### **Elbow Method**
        - Plots the *inertia* (sum of squared distances from points to their assigned cluster centers) for different values of k.
        - **Interpretation:**  
          Inertia always decreases as k increases (clusters fit the data better).
          The "elbow" point, where inertia drops off more slowly, suggests a good balance: adding more clusters past this point yields diminishing returns.

        ### **Silhouette Score**
        - Measures how similar a store is to its assigned cluster versus other clusters, ranging from -1 (bad) to +1 (good).
        - **Interpretation:**  
          Higher values indicate better-defined, more separated clusters.
          If silhouette scores drop as k increases, extra clusters are likely splitting natural groups unnecessarily.

        **Combined approach:**  
        Use both metrics together!
        - The elbow in the inertia plot indicates the minimal effective k.  
        - The silhouette score confirms if clusters are cohesive and well-separated.
        """)
    inertias = []
    silhouettes = []
    ks = range(2, 7)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sil_score = silhouette_score(X, km.labels_)
        silhouettes.append(sil_score)
    st.write(pd.DataFrame({"k": list(ks), "Inertia": inertias, "Silhouette": silhouettes}))

    fig_k, ax_k = plt.subplots(1, 2, figsize=(12, 4))
    ax_k[0].plot(list(ks), inertias, 'o-', label='Inertia')
    ax_k[0].set_title('Elbow Method')
    ax_k[0].set_xlabel('k')
    ax_k[0].set_ylabel('Inertia')
    ax_k[0].grid(True)
    ax_k[1].plot(list(ks), silhouettes, 'o-', label='Silhouette')
    ax_k[1].set_title('Silhouette Scores')
    ax_k[1].set_xlabel('k')
    ax_k[1].set_ylabel('Silhouette Score')
    ax_k[1].grid(True)
    st.pyplot(fig_k)

    st.markdown("""
     #### **Summary Table:**
| k | Inertia | Silhouette | Interpretation |
|---|---------|------------|---------------|
| 2 |  111    | 0.35       | Big, coarse groups. Simple, but may hide differences. |
| 3 |   84    | 0.34       | Reasonable compromise; interpretable clusters. |
| 4 |   65    | 0.33       | Still interpretable, with more detail. |
| 5 |   51    | 0.38       | Best separation, but more segments to manage. |
| 6 |   41    | 0.36       | Over-segmentation risk. |           

**Interpreting the Elbow and Silhouette Plots**  
- The inertia curve drops steeply from k=2 to k=3, then more slowly for k>3.
- The "elbow" (where the drop starts to flatten) typically indicates the optimal k. In your plot, the elbow appears around **k=3 or k=4**. Beyond this, inertia continues to decrease but much less dramatically, meaning extra clusters are not providing major improvement in cluster compactness.
- The silhouette score starts around **0.35** at k=2 and **drops slightly** at k=3 and k=4, then **increases again** at k=5 and k=6.
- A higher silhouette score indicates that points are well-matched to their own cluster and well-separated from others.
- In our plot, **k=5 gives the highest silhouette score (~0.38)**, slightly higher than other values, but the difference is not dramatic.
- **k=3 or k=4** is a reasonable compromise for interpretability and simplicity.

*How do we decide?**
- **k=3 or k=4**:  
  - Offers a simple segmentation with clear differences in inertia, and is easy to interpret.  
  - Business context may favor fewer, larger segments.
- **k=5 or k=6**:  
  - Gives slightly higher silhouette scores, indicating more cohesive clusters.  
  - However, more clusters may lead to over-segmentation and less actionable business groups.
- **Compromise:**  
  - If you want easy business interpretation, **k=3 or k=4** is often preferred—these are supported by the "elbow."
  - If you need the absolute best cluster separation for, say, targeted marketing, and can handle more segments, **k=5** is mathematically optimal here.
    """)

    # ============================================================
    # 7. K-MEANS CLUSTERING & CLUSTER PROFILES
    # ============================================================
    st.write("---")
    st.header("7️) K-Means Clustering & Cluster Profiles", anchor="k-means-clustering--cluster-profiles")
    st.markdown("""
    - **Chosen k = 3** based on elbow/silhouette tradeoff.
    - Fit model and compute cluster-level averages on original scale for interpretability.
    """)
    n_kmeans = 3
    km = KMeans(n_clusters=n_kmeans, random_state=42)
    clean['cluster'] = km.fit_predict(X)
    profiles = clean.groupby('cluster').agg(
        count=('Store','count'),
        avg_sales=('avg_sales','mean'),
        avg_fuel=('avg_fuel','mean'),
        avg_CPI=('avg_CPI','mean'),
        avg_unemp=('avg_unemp','mean')
    ).reset_index()
    def format_sales(val):
        val = float(val)
        if val >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        elif val >= 1_000:
            return f"${val/1_000:.0f}K"
        else:
            return f"${val:,.0f}"
    profiles_display = profiles.copy()
    profiles_display['avg_sales'] = profiles_display['avg_sales'].apply(format_sales)
    profiles_display['avg_fuel'] = profiles_display['avg_fuel'].round(2)
    profiles_display['avg_CPI'] = profiles_display['avg_CPI'].round(1)
    profiles_display['avg_unemp'] = profiles_display['avg_unemp'].round(1)
    st.dataframe(profiles_display.style.set_caption("Cluster Profiles (Readable: $M/$K)"))
    st.markdown("#### Cluster Profiles: Radar Chart Comparison")
    feature_names = ['avg_sales','avg_fuel','avg_CPI','avg_unemp']
    feature_labels = [
        "Avg Sales ($K)", "Avg Fuel ($)", "Avg CPI", "Avg Unemp (%)"
    ]
    vmin = profiles[feature_names].min().min() * 0.97
    vmax = profiles[feature_names].max().max() * 1.03
    colors = cm.get_cmap('Set1', profiles.shape[0])
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]
    fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i in range(profiles.shape[0]):
        values = [profiles.iloc[i][col] for col in feature_names]
        values += values[:1]
        ax_radar.plot(angles, values, label=f"Cluster {i}", color=colors(i), linewidth=2)
        ax_radar.fill(angles, values, color=colors(i), alpha=0.10)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feature_labels, fontsize=13, fontweight='bold')
    ax_radar.set_yticklabels([])
    ax_radar.set_title('All Cluster Profiles (Radar Comparison)', size=16, pad=22)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    st.pyplot(fig_radar)

    # ============================================================
    # 8. CLUSTER STABILITY: ASSESSING CLUSTER SIZE VARIABILITY
    # ============================================================
    st.write("---")
    st.header("8️⃣ Cluster Stability: Assessing Cluster Size Variability", anchor="cluster-stability-assessing-cluster-size-variability")
    st.markdown("""
    K-Means can yield slightly different cluster assignments depending on its random initialization.  
    If the algorithm is unstable, cluster sizes (how many stores per group) will fluctuate a lot from run to run.
    """)
    with st.expander("Show Cluster Stability Explanation"):
        st.markdown("""
        - We re-run K-Means clustering 10 times, each with a different random seed.
        - For each run, we count how many stores end up in each cluster.
        - We summarize the distribution of cluster sizes across runs (mean, std, min, max, quartiles).

        **Why check cluster size variability?**  
        K-Means can yield slightly different cluster assignments depending on its random initialization. If the algorithm is unstable, cluster sizes (how many stores per group) will fluctuate a lot from run to run.

        **How is this tested?**  
        - We re-run K-Means clustering 10 times, each with a different random seed.
        - For each run, we count how many stores end up in each cluster.
        - We summarize the distribution of cluster sizes across runs (mean, std, min, max, quartiles).

        **How to interpret the table:**  
        - **Mean**: Average number of stores per cluster over 10 runs.
        - **Std**: Standard deviation; higher values mean more instability in assignments.
        - **Min/Max**: Smallest/largest observed cluster size over the 10 runs.
        - **Quartiles (25%, 50%, 75%)**: Show the spread of typical sizes.

        **Ideal result:**  
        - **Low standard deviation (std)** means cluster assignments are stable and not sensitive to random starting conditions.
        - If the **min/max** sizes differ a lot, or if the std is high, that suggests cluster boundaries are fuzzy or poorly separated.

        **What we see here:**  
        - The means are close to each other (e.g., ~15 for each cluster).
        - Standard deviations are moderate (around 5-6 stores), and min/max show some spread.
        - This is reasonable stability, but not perfect—some stores may lie near cluster boundaries and occasionally switch group depending on initialization.

        **Bottom line:**  
        - These results support that our clusters are generally robust, but further increases in k, or adding/removing features, could change cluster assignment for a small number of stores.
        """)
    cluster_counts = []
    for seed in range(10):
        km_test = KMeans(n_clusters=n_kmeans, random_state=seed).fit(X)
        counts = pd.Series(km_test.labels_).value_counts().sort_index()
        cluster_counts.append(counts.values)
    stability_df = pd.DataFrame(cluster_counts, columns=[f"Cluster_{i}" for i in range(n_kmeans)])
    st.dataframe(stability_df.describe().transpose())

    # ============================================================
    # 9. PCA FOR VISUALIZATION & INSIGHT
    # ============================================================
    st.write("---")
    st.header("9️⃣ PCA for Visualization & Insight", anchor="pca-for-visualization--insight")
    st.markdown("""
    Principal Component Analysis (PCA) helps reduce the dimensionality of our data (from 4 features to 2 principal components), making it possible to visualize store clusters in a simple scatter plot. PCA finds the linear combinations of the original features that capture the greatest variance. Here, we plot each store by its position along the first two principal components, colored by its cluster assignment.
    """)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    fig_pca, ax_pca = plt.subplots(figsize=(7, 6))
    scatter = ax_pca.scatter(pcs[:,0], pcs[:,1], c=clean['cluster'], cmap='tab10', s=60)
    ax_pca.set_xlabel('PC1')
    ax_pca.set_ylabel('PC2')
    ax_pca.set_title('Clusters in PCA Space')
    legend1 = ax_pca.legend(*scatter.legend_elements(), title='Cluster')
    ax_pca.add_artist(legend1)
    ax_pca.grid(True)
    st.pyplot(fig_pca)
    ev = pd.DataFrame({'PC': ['PC1','PC2'], 'ExplainedVar': pca.explained_variance_ratio_})
    st.write("Explained variance per PC:")
    st.dataframe(ev)
    loadings = pd.DataFrame(pca.components_.T, index=features.columns, columns=['PC1','PC2'])
    st.write('PCA Loadings (contribution of each feature to each PC):')
    st.dataframe(loadings)
    loadings_pct = loadings.abs().div(loadings.abs().sum(axis=0), axis=1) * 100
    st.write('PCA Loadings (% contribution to each PC):')
    st.dataframe(loadings_pct.round(2))
    st.write("""
    ### Output Explanation: PCA Scatter & Variance Table

    - **PCA Scatter Plot:** Each dot represents a store, colored by cluster. Distinct groupings suggest effective clustering and that PCA captures meaningful variance for cluster separation.
    - **Explained Variance Table:** Shows what proportion of the total variance each PC captures (e.g., PC1 ≈ 51%, PC2 ≈ 26%).
    - **PCA Loadings:** These are the weights (contributions) of each original feature to the PCs. Larger absolute values indicate more influence on that principal component.

    #### Output Explanation: Loadings and Centroids

    - **% Loadings:** For each principal component (PC), these percentages show how much each feature contributes. For example, PC1 is mostly driven by `avg_fuel` and `avg_CPI`, while PC2 is dominated by `log_sales`.
    - **Cluster Centroids:** Each row is a cluster’s center in the original features, allowing for practical understanding of typical sales, fuel prices, CPI, and unemployment for each group.
    """)
    centers_scaled = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_orig, columns=features.columns)
    st.write('Cluster Centroids (original feature scale):')
    st.dataframe(centers_df.round(2))

    # ============================================================
    # 10. NUMERIC CLUSTER SUMMARY & BAR CHARTS
    # ============================================================
    st.write("---")
    st.header("10) Numeric Cluster Summary & Bar Charts", anchor="numeric-cluster-summary--bar-charts")
    st.markdown("""
    Each row is a cluster. You can see cluster sizes, typical sales (in raw dollars), fuel price, CPI, and unemployment rate.
    """)
    st.dataframe(profiles_display[['cluster','count','avg_sales','avg_fuel','avg_CPI','avg_unemp']])

    st.markdown("""
    By combining PCA visualizations, loadings, and numeric profiles, we ensure both statistical validity (good separation, meaningful PCs) and business relevance (clusters are interpretable in the original feature units).  
    Next, we’ll explore these differences visually and discuss what actions they suggest for each cluster type.
    """)

    # ============================================================
    # 11. INTERPRETATION & BUSINESS INSIGHTS
    # ============================================================
    st.write("---")
    st.header("1️⃣1️⃣ Interpretation & Business Insights", anchor="interpretation--business-insights")
    st.markdown("""
    Below, the cluster numbers, store counts, and statistics **directly match the output above**.  
    All dollar values are formatted for clarity (e.g., "$1.28M" for 1,280,700).

    ### Cluster Profiles & Business Actions

    **Cluster 0: High-Performing, Stable**  
    - **Number of Stores:** See cluster profile table  
    - **Interpretation:** Top-performing stores with strong, stable sales, operating in moderate economic environments.  
    - **Recommended Action:**  
        - Maintain and reinforce best practices.
        - Pilot premium services or exclusive product lines.
        - Invest in loyalty and customer experience initiatives.

    **Cluster 1: Lower Sales, Higher Economic Pressure**  
    - **Number of Stores:** See cluster profile table  
    - **Interpretation:** In tougher markets, facing higher unemployment and likely more price-sensitive customers, resulting in lower sales.  
    - **Recommended Action:**  
        - Launch targeted promotions, discounts, and community-based campaigns.
        - Strengthen local partnerships and support programs.
        - Consider operational efficiency improvements to protect margins.

    **Cluster 2: Moderate Sales, High Price Index**  
    - **Number of Stores:** See cluster profile table  
    - **Interpretation:** Stores with moderate sales, but located in high-CPI (costlier) markets.  
    - **Recommended Action:**  
        - Test value-based promotions or competitive pricing on essentials.
        - Localize marketing to match demographic and economic realities.
        - Explore partnerships or events that emphasize value and community.

    ---
    **Practical Implications:**  
    - **Operations:** Prioritize resource allocation (inventory, staff, budget) by cluster needs and expected sales demand.
    - **Marketing:** Customize campaigns by cluster.
    - **Finance & Forecasting:** Use clusters to generate scenario-based sales forecasts and set realistic, segment-specific financial goals.

    ---
    > **Summary:**  
    > Clustering enables actionable, data-driven segmentation. By understanding the economic and sales environment of each cluster, Walmart can deploy more precise, effective strategies to grow sales, optimize operations, and better serve local communities.
    """)

    # ============================================================
    # 12. EXPANDED MEAN-SHIFT CLUSTERING IMPLEMENTATION
    # ============================================================
    st.write("---")
    st.header("1️2) Expanded Mean-Shift Clustering Implementation", anchor="expanded-mean-shift-clustering-implementation")
    st.markdown("""
    **Why Mean-Shift?**  
    Unlike K-Means, Mean-Shift clustering does **not require pre-specifying the number of clusters (k)**. Instead, it automatically discovers the number of distinct groups based on the density of data points. This approach is useful for exploratory analysis where the true structure of the data is unknown.

    - **Step 1:** Scale features for fair clustering.
    - **Step 2:** Run Mean-Shift to determine clusters.
    - **Step 3:** Profile and visualize clusters, just as before.
    - **Step 4:** Summarize actionable insights from these new, data-driven segments.
    """)
    features_ms = clean[['avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']]
    scaler_ms = StandardScaler()
    X_scaled_ms = scaler_ms.fit_transform(features_ms)
    bandwidth = estimate_bandwidth(X_scaled_ms, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X_scaled_ms)
    labels_ms = ms.labels_
    n_clusters_ms = len(np.unique(labels_ms))
    st.write(f"Estimated number of clusters by Mean-Shift: {n_clusters_ms}")
    clean['ms_cluster'] = labels_ms
    centers_scaled_ms = ms.cluster_centers_
    centers_orig_ms = scaler_ms.inverse_transform(centers_scaled_ms)
    centers_df_ms = pd.DataFrame(centers_orig_ms, columns=features_ms.columns)
    st.write('Mean-Shift Cluster Centroids (original feature scale):')
    st.dataframe(centers_df_ms.round(2))
    pca_ms = PCA(n_components=2, random_state=42)
    proj_ms = pca_ms.fit_transform(X_scaled_ms)
    fig_ms, ax_ms = plt.subplots(figsize=(7, 6))
    scatter_ms = ax_ms.scatter(
        proj_ms[:, 0], proj_ms[:, 1], c=labels_ms, cmap='tab10', alpha=0.7, s=80, edgecolor='k', linewidth=0.5
    )
    ax_ms.set_title('MeanShift Clusters (PCA Projection)', fontsize=15)
    ax_ms.set_xlabel('PC1')
    ax_ms.set_ylabel('PC2')
    handles, legend_labels = scatter_ms.legend_elements(prop="colors")
    legend_labels = [f"Cluster {i}" for i in np.unique(labels_ms)]
    ax_ms.legend(handles, legend_labels, title="Cluster", loc='best', fontsize=11)
    st.pyplot(fig_ms)
    profiles_ms = (
        clean
        .groupby('ms_cluster')
        .agg(
            count=('Store', 'count'),
            avg_sales=('avg_sales', 'mean'),
            avg_fuel=('avg_fuel', 'mean'),
            avg_CPI=('avg_CPI', 'mean'),
            avg_unemp=('avg_unemp', 'mean')
        )
        .reset_index()
    )
    profiles_ms_display = profiles_ms.copy()
    profiles_ms_display['avg_sales'] = profiles_ms_display['avg_sales'].apply(format_sales)
    profiles_ms_display['avg_fuel'] = profiles_ms_display['avg_fuel'].round(2)
    profiles_ms_display['avg_CPI'] = profiles_ms_display['avg_CPI'].round(1)
    profiles_ms_display['avg_unemp'] = profiles_ms_display['avg_unemp'].round(1)
    st.write('Mean-Shift Cluster Profiles:')
    st.dataframe(profiles_ms_display[['ms_cluster', 'count', 'avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']])
    st.markdown("""
    #### Output: Mean-Shift Cluster Profiles

    - **Cluster**: Cluster number assigned by Mean-Shift.
    - **Count**: Number of stores in each cluster.
    - **Avg Sales**: Average weekly sales (formatted for readability).
    - **Avg Fuel, CPI, Unemployment**: Economic context for each group.

    These profiles are useful for targeting business strategies to each segment.
    """)

    st.markdown("""
    ### Interpretation: K-Means vs. Mean-Shift Clustering

Both K-Means and Mean-Shift algorithms partition stores based on sales and economic features, but they approach the problem differently:

#### K-Means Clustering:
- **Requires pre-specifying the number of clusters (k).**
- **Finds clusters of roughly similar size** and is best when clusters are spherical and evenly distributed.
- **Actionability:** Useful for standard segmentation (e.g., "Top", "Middle", "Vulnerable" stores) and scenario analysis.

#### Mean-Shift Clustering:
- **Automatically determines the number of clusters** based on underlying data density.
- **Can detect clusters of varying sizes and shapes,** especially if the data distribution is non-uniform.
- **Actionability:** Reveals "natural" groupings that may highlight previously unnoticed store segments (e.g., outlier high-performers or small specialized groups).

### Comparing Results:
- **Cluster Counts:** Mean-Shift may yield more or fewer clusters than K-Means, depending on the density structure.
- **Cluster Profiles:** Some clusters may overlap with those found by K-Means, but others may isolate niche or transitional store types.
- **Business Strategy:** Use insights from both methods for robust decision-making—K-Means for consistency and tracking, Mean-Shift for uncovering hidden opportunities or risks.

- **Cluster 0:**  
  - *Profile:* [Summarize key features, e.g., moderate sales, high CPI]  
  - *Action:* Targeted marketing in high-cost areas, adapt product mix to local preferences.

- **Cluster 1:**  
  - *Profile:* [Summarize key features, e.g., high sales, low unemployment]  
  - *Action:* Maintain premium positioning, pilot new product launches, leverage local economic stability.

- **Cluster 2:**  
  - *Profile:* [Summarize key features, e.g., low sales, high fuel prices]  
  - *Action:* Increase cost-efficiency, focus on promotions tied to fuel price fluctuations.

- **Cluster 3:**  
  - *Profile:* [Summarize key features, e.g., small group, unique combination of metrics]  
  - *Action:* Investigate further—these may be special cases worth deeper qualitative analysis.

    """)

    # ============================================================
    # 13. CONCLUSION
    # ============================================================
    st.header("Conclusion", anchor="conclusion") 

    st.markdown("""
   
    **Research Question 2 (RQ2):**  
    *Are there natural groups of Walmart stores based on sales and economic characteristics?*

    **Findings:**  
    We identified **three distinct store clusters**, each reflecting unique sales behaviors and sensitivities to local economic conditions.

    **Hypothesis Validation:**  
    - **Supported** — The clusters align with hypothesized business types:  
        - “Stable/High-Performing”  
        - “Price-Sensitive/Economically Pressured”  
        - “Moderate/High-Cost Environment”

    1. **Data Aggregation & Cleaning:**  
       Careful aggregation to the store level and robust cleaning ensured each data point represented a meaningful, comparable business unit.
    2. **Outlier Removal:**  
       Excluding extreme outlier stores allowed for more representative and interpretable clustering. (Note: Outliers may warrant targeted, separate strategy.)
    3. **Log Transformation & Scaling:**  
       These steps reduced skewness and balanced feature influence, making clustering results robust and fair.
    4. **K-Means (k=3):**  
       The algorithm revealed three actionable clusters, each with clear operational and strategic implications for Walmart.
    5. **PCA Visualization & Feature Insight:**  
       PCA made clusters visually interpretable and confirmed which features drive the main axes of store variation.

    ---

    > **Final Insight:**  
    > This data-driven clustering approach provides actionable segmentation. By understanding store clusters and their economic context, Walmart can make smarter, tailored decisions in marketing, operations, and long-term planning.
    """)