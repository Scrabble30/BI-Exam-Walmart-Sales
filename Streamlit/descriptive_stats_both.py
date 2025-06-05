# streamlit_app.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import iqr, skew, kurtosis, levene

def load_data():
    cleaned = pd.read_csv("../Data/Cleaned-Walmart_Sales.csv", parse_dates=['Date'], dayfirst=True)
    raw = pd.read_csv("../Data/Walmart_Sales.csv", parse_dates=['Date'], dayfirst=True)
    return cleaned, raw

def descriptive_stats(sales):
    return {
        'mean': sales.mean(),
        'median': sales.median(),
        'min': sales.min(),
        'max': sales.max(),
        'range': sales.max() - sales.min(),
        'std_dev': sales.std(),
        'iqr': iqr(sales),
        'skewness': skew(sales),
        'kurtosis': kurtosis(sales)
    }

def stats_table(df, title):
    st.subheader(title)
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Holiday Week Statistics**")
        st.json(descriptive_stats(holiday_sales))
    with col2:
        st.markdown("**Non-Holiday Week Statistics**")
        st.json(descriptive_stats(non_holiday_sales))

    stat, p = levene(holiday_sales, non_holiday_sales)
    st.write(f"**Levene's Test statistic**: {stat:.4f}, **p-value**: {p:.4f}")
    if p < 0.05:
        st.success("✅ Significant difference in variance between holiday and non-holiday weeks.")
    else:
        st.error("❌ No significant difference in variance between holiday and non-holiday weeks.")

def plot_distributions(df, label):
    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']

    st.markdown(f"### Boxplot ({label})")
    fig, ax = plt.subplots()
    sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df, ax=ax)
    ax.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax.set_title(f'Boxplot: Weekly Sales ({label})')
    st.pyplot(fig)

    st.markdown(f"### KDE Histogram ({label})")
    fig2, ax2 = plt.subplots()
    sns.histplot(holiday_sales, kde=True, color='green', label='Holiday', ax=ax2, stat='density', bins=30)
    sns.histplot(non_holiday_sales, kde=True, color='blue', label='Non-Holiday', ax=ax2, stat='density', bins=30)
    ax2.set_title(f'Sales Distribution: {label}')
    ax2.legend()
    st.pyplot(fig2)

def main():
    st.title("Holiday vs. Non-Holiday Weekly Sales - Raw vs. Cleaned Dataset")
    st.markdown("---")
    
    st.write("Kurtosis > 0 → Leptokurtic: More outliers")
    st.write("Kurtosis ≈ 0 → Mesokurtic: Moderate outliers")
    st.write("Kurtosis < 0 → Platykurtic: Fewer outliers")

    st.markdown("---")

    cleaned_df, raw_df = load_data()

    st.header("Cleaned Dataset Analysis")
    stats_table(cleaned_df, "Descriptive Statistics (Cleaned)")
    plot_distributions(cleaned_df, "Cleaned Data")
    st.markdown("---")

    st.write("**Why do i think this might have not been the right approach:**")
    st.write("By removing outliers (e.g., Black Friday, Christmas), we're possibly also removed the very sales spikes that define the volatility of holidays")

    st.markdown("---")
    st.header("Raw Dataset Analysis")
    stats_table(raw_df, "Descriptive Statistics (Raw)")
    plot_distributions(raw_df, "Raw Data")
    st.markdown("---")

    st.write("Using the Raw data we can understand true business impact, because it include big events like blackfriday and such ")
    st.write("And this shows significant difference in variance, supporting our hypothesis that holiday weeks show higher sales variability")


if __name__ == "__main__":
    main()
