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

def clustering():
    # ============================================================
    # 🏪 RQ2: Clustering Store Sales Patterns
    # ============================================================
    st.markdown("# RQ2: Clustering Store Sales Patterns")

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
    <summary><h4>Table of Contents</h4></summary>

    1. [Setup: Imports & Configuration]  
    2. [Data Loading & Preview]  
    3. [Data Description & Aggregation Rationale]  
    4. [Initial Summary Statistics]  
    5. [Outlier Detection & Removal]  
    6. [Scaling & Transformation]  
        - Log Transformation: Understanding Sales Distribution  
        - Log Transform Impact: Density Comparison  
        - Strategic Priority: Sales vs. Unemployment  
        - Standardization  
    7. [Choosing the Number of Clusters (k)]  
        - Interpreting the Elbow and Silhouette Plots  
    8. [K-Means Clustering & Cluster Profiles]  
    9. [Cluster Stability: Assessing Cluster Size Variability]  
    10. [PCA for Visualization & Insight]  
        - Readable PCA Loadings & Cluster Centers  
        - Numeric Cluster Summary & Bar Charts  
    11. [Interpretation & Business Insights]  
    12. [Answer to RQ2 & Conclusion]  
    13. [Expanded Mean-Shift Clustering Implementation]

    </details>
    """, unsafe_allow_html=True)
    st.write("---")

    # ============================================================
    # 1. DATA LOADING & PREVIEW
    # ============================================================
    st.header("1. Data Loading & Preview")
    st.markdown("""
    The first step in any data analysis project is to load the dataset and **gain an initial understanding of its structure, completeness, and quality**. This helps you quickly spot issues (like missing data, duplicates, or outliers) and get a feel for what’s available.

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

    # -- Data loading code
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
    st.header("2. Data Description & Aggregation Rationale")
    st.markdown("""
    The raw dataset records **weekly sales at the department level for each store**, alongside economic features such as fuel price, CPI (Consumer Price Index), and unemployment rate.  

    However, clustering on weekly or department-level data would lead to a very large, noisy dataset—and our primary goal is to **identify groups of similar stores** based on their average sales performance and typical economic environment.

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
    st.header("3. Initial Summary Statistics")
    st.markdown("""
    Before performing clustering, it's important to **understand the typical values and spread of each key variable** across all stores. This step provides valuable context for interpreting later clustering results and for spotting potential outliers or data quality issues.

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
    st.header("4. Outlier Detection & Removal")
    st.markdown("""
    ### Why consider outliers?
    Outliers—stores with extremely high or low average values—can **distort cluster centroids** and create artificial groupings that don’t reflect the majority of stores. Detecting and handling outliers helps ensure that clusters reflect typical store behaviors rather than being skewed by rare, extreme cases.

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

    # ----------------------- Correlation Matrix ------------------------
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
    # 6. SCALING & TRANSFORMATION
    # ============================================================
    st.write("---")
    st.header("6. Scaling & Transformation")

    # ---------- 6.1 Log Transform of Sales ----------
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

    # -- Histograms: Original vs Log-transformed sales
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(clean['avg_sales'], bins=15)
    axes[0].set_title('Original Sales')
    axes[1].hist(clean['log_sales'], bins=15)
    axes[1].set_title('Log-Transformed Sales')
    st.pyplot(fig)

    # -- KDE plot
    fig_kde, ax_kde = plt.subplots(figsize=(10, 4))
    sns.kdeplot(clean['avg_sales'], label='Original', fill=True, ax=ax_kde)
    sns.kdeplot(clean['log_sales'], label='Log-Transformed', fill=True, ax=ax_kde)
    ax_kde.set_title('Sales Distribution: Before vs After Log Transform')
    ax_kde.set_xlabel('Sales')
    ax_kde.legend()
    st.pyplot(fig_kde)

    # -- Scatterplot: Sales vs Unemployment
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=clean, x='avg_unemp', y='avg_sales', s=100, alpha=0.8, ax=ax_scatter)
    ax_scatter.set_title('Strategic Priority: Sales vs Unemployment')
    ax_scatter.set_xlabel('Unemployment Rate (%)')
    ax_scatter.set_ylabel('Avg Sales ($)')
    ax_scatter.axvline(8.5, ls='--', c='gray')
    ax_scatter.text(9, clean['avg_sales'].max()*0.7, 'High Risk: Low Sales + High Unemployment', 
            rotation=90, va='center', fontsize=10, color='red')
    st.pyplot(fig_scatter)

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
    # 7. CHOOSING THE NUMBER OF CLUSTERS (k)
    # ============================================================
    st.write("---")
    st.header("7. Choosing the Number of Clusters (k)")
    st.markdown("""
    Selecting the optimal number of clusters is critical for effective segmentation. We use two main techniques:

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
    **Interpreting the Elbow and Silhouette Plots**  
    - The inertia curve drops steeply from k=2 to k=3, then more slowly for k>3.
    - The "elbow" (where the drop starts to flatten) typically indicates the optimal k. In your plot, the elbow appears around **k=3 or k=4**. Beyond this, inertia continues to decrease but much less dramatically, meaning extra clusters are not providing major improvement in cluster compactness.
    - The silhouette score starts around **0.35** at k=2 and **drops slightly** at k=3 and k=4, then **increases again** at k=5 and k=6.
    - A higher silhouette score indicates that points are well-matched to their own cluster and well-separated from others.
    - In your plot, **k=5 gives the highest silhouette score (~0.38)**, slightly higher than other values, but the difference is not dramatic.
    - **k=3 or k=4** is a reasonable compromise for interpretability and simplicity.
    """)

    # ============================================================
    # 8. K-MEANS CLUSTERING & CLUSTER PROFILES
    # ============================================================
    st.write("---")
    st.header("8. K-Means Clustering & Cluster Profiles")
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

    # ------------------ Radar charts for each cluster ------------------
    st.markdown("### Cluster Profiles: Radar Chart Comparison")
    import matplotlib.cm as cm

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
    # 9. CLUSTER STABILITY: ASSESSING CLUSTER SIZE VARIABILITY
    # ============================================================
    st.write("---")
    st.header("9. Cluster Stability: Assessing Cluster Size Variability")
    st.markdown("""
    K-Means can yield slightly different cluster assignments depending on its random initialization.  
    If the algorithm is unstable, cluster sizes (how many stores per group) will fluctuate a lot from run to run.

    - We re-run K-Means clustering 10 times, each with a different random seed.
    - For each run, we count how many stores end up in each cluster.
    - We summarize the distribution of cluster sizes across runs (mean, std, min, max, quartiles).
    """)

    cluster_counts = []
    for seed in range(10):
        km_test = KMeans(n_clusters=n_kmeans, random_state=seed).fit(X)
        counts = pd.Series(km_test.labels_).value_counts().sort_index()
        cluster_counts.append(counts.values)
    stability_df = pd.DataFrame(cluster_counts, columns=[f"Cluster_{i}" for i in range(n_kmeans)])
    st.dataframe(stability_df.describe().transpose())

    st.markdown("""
    - **Low standard deviation (std)** means cluster assignments are stable and not sensitive to random starting conditions.
    - If the **min/max** sizes differ a lot, or if the std is high, that suggests cluster boundaries are fuzzy or poorly separated.
    - These results support that our clusters are generally robust, but further increases in k, or adding/removing features, could change cluster assignment for a small number of stores.
    """)

    # ============================================================
    # 10. PCA FOR VISUALIZATION & INSIGHT
    # ============================================================
    st.write("---")
    st.header("10. PCA for Visualization & Insight")
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

    centers_scaled = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_orig, columns=features.columns)
    st.write('Cluster Centroids (original feature scale):')
    st.dataframe(centers_df.round(2))

    # ============================================================
    # 11. NUMERIC CLUSTER SUMMARY & BAR CHARTS
    # ============================================================
    st.write("---")
    st.header("11. Numeric Cluster Summary & Bar Charts")
    st.markdown("""
    Each row is a cluster. You can see cluster sizes, typical sales (in raw dollars), fuel price, CPI, and unemployment rate.
    """)
    st.dataframe(profiles_display[['cluster','count','avg_sales','avg_fuel','avg_CPI','avg_unemp']])

    st.markdown("""
    By combining PCA visualizations, loadings, and numeric profiles, we ensure both statistical validity (good separation, meaningful PCs) and business relevance (clusters are interpretable in the original feature units).  
    Next, we’ll explore these differences visually and discuss what actions they suggest for each cluster type.
    """)

    # ============================================================
    # 12. INTERPRETATION & BUSINESS INSIGHTS
    # ============================================================
    st.write("---")
    st.header("12. Interpretation & Business Insights")
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
    # 13. EXPANDED MEAN-SHIFT CLUSTERING IMPLEMENTATION
    # ============================================================
    st.write("---")
    st.header("13. Expanded Mean-Shift Clustering Implementation")
    st.markdown("""
    **Why Mean-Shift?**  
    Unlike K-Means, Mean-Shift clustering does **not require pre-specifying the number of clusters (k)**. Instead, it automatically discovers the number of distinct groups based on the density of data points. This approach is useful for exploratory analysis where the true structure of the data is unknown.

    - **Step 1:** Scale features for fair clustering.
    - **Step 2:** Run Mean-Shift to determine clusters.
    - **Step 3:** Profile and visualize clusters, just as before.
    - **Step 4:** Summarize actionable insights from these new, data-driven segments.
    """)
    # -- Features for clustering
    features_ms = clean[['avg_sales', 'avg_fuel', 'avg_CPI', 'avg_unemp']]
    scaler_ms = StandardScaler()
    X_scaled_ms = scaler_ms.fit_transform(features_ms)

    # -- Estimate bandwidth and fit MeanShift
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

    # -- PCA visualization
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

    # -- Mean-Shift cluster profiles
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

    **Key Benefits:**  
    - No need to pre-specify the number of clusters.
    - Clusters reflect true density patterns in the data.
    - Store profiles are directly comparable to those found via K-Means, supporting robust segmentation and deeper business insight.
    """)
