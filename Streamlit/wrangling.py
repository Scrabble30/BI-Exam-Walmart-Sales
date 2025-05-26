import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def walmart_cleanup():
    st.title("Walmart Sales – Data Cleanup Overview")

    # Load cleaned and raw (before outlier removal) datasets
    df_cleaned = pd.read_csv("../Data/Cleaned-Walmart_Sales.csv")
    df_raw = pd.read_csv("../Data/Walmart_Sales.csv").dropna()

    st.markdown("""
    <div style="border: 0px solid #ccc; padding: 15px; border-radius: 8px; background-color: black; color: white;">
    <h3>What We Did</h3>

    This page explains how we cleaned and prepared the <strong>Walmart Sales</strong> dataset for analysis.

    ---

    ### Key Steps

    1. **Loaded the raw CSV** and dropped rows with missing data.
    2. **Checking constant columns** We looked for constant clumns with a unique value, but there was none.
    3. **Removed outliers** using the IQR method:
        - For each numerical column:
            - Calculated Q1, Q3
            - Removed values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
    4. **Saved cleaned dataset** as `Cleaned-Walmart_Sales.csv`.

    ---

    ### Goal

    To remove noise and prepare the dataset for modeling and visualization.

    </div>
    """, unsafe_allow_html=True)

    # Show a preview of the cleaned data
    st.subheader("Preview of Cleaned Data")
    st.dataframe(df_cleaned.head(10))

    # Outlier comparison section
    st.header("Q–Q Plots: Before vs After Outlier Removal")

    # Select numerical columns (excluding Holiday_Flag)
    numeric_cols = df_cleaned.select_dtypes(include=["float64", "int64"]).columns.difference(["Holiday_Flag"])
    selected_col = st.selectbox("Select a column to view Q–Q plots", options=numeric_cols)

    # Generate Q–Q plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    stats.probplot(df_raw[selected_col], dist="norm", plot=axs[0])
    axs[0].set_title(f"Before - Q–Q Plot of {selected_col}")

    stats.probplot(df_cleaned[selected_col], dist="norm", plot=axs[1])
    axs[1].set_title(f"After - Q–Q Plot of {selected_col}")

    st.pyplot(fig)

    # Summary stats
    st.markdown(f"""
    **Rows before cleaning:** {len(df_raw)}  
    **Rows after outlier removal:** {len(df_cleaned)}  
    **Total rows removed:** {len(df_raw) - len(df_cleaned)}
    """)
