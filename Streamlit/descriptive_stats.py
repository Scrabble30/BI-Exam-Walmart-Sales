import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import iqr, skew, kurtosis, levene

def descriptivestatistics():
    st.title("RQ 4 - Descriptive Statistics")
    
    st.markdown("""
    <div style="border:0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ4: Descriptive Statistics</strong></li>
            <li><strong>Question:</strong> What are the distribution patterns (mean, median, range, standard deviation) of weekly sales for holidays vs. non-holidays?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>H:</strong> Holiday weeks show significantly higher variance in weekly sales compared to non-holiday weeks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("We are working with the cleaned Walmart Sales dataset where outliers have been removed for better precision.")

    # Load data
    directory = "../Data/"
    file_name = "Cleaned-Walmart_Sales.csv"
    df = pd.read_csv(directory + "/" + file_name, parse_dates=['Date'], dayfirst=True)

    # Display basic info
    if st.checkbox("Show data preview"):
        st.dataframe(df.head())

    st.write("----")

    st.subheader("Step 1: Holiday vs Non-Holiday Distribution")

    # Bar chart of Holiday vs Non-Holiday
    st.write("We start by examining how many holiday and non-holiday weeks are present in the dataset.")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Holiday_Flag", palette='pastel', ax=ax1)
    ax1.set_xticklabels(['Not Holiday', 'Holiday'])
    ax1.set_title('Count of Observations by Holiday Flag')
    st.pyplot(fig1)

    # Pie Chart
    fig2, ax2 = plt.subplots()
    ax2.pie(df['Holiday_Flag'].value_counts(), labels=['Not Holiday', 'Holiday'],
            autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    ax2.set_title('Distribution of Holiday Flag')
    st.pyplot(fig2)

    st.write("----")

    st.subheader("Step 2: Descriptive Statistics of Weekly Sales")

    holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales']
    non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales']

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

    holiday_stats = descriptive_stats(holiday_sales)
    non_holiday_stats = descriptive_stats(non_holiday_sales)

    st.write("We calculate the following metrics: **mean**, **median**, **range**, **standard deviation**, **IQR**, **skewness**, and **kurtosis**.")

    st.markdown("#### Holiday Week Statistics")
    st.json(holiday_stats)

    st.markdown("#### Non-Holiday Week Statistics")
    st.json(non_holiday_stats)

    st.write("----")

    st.subheader("Step 3: Visualizing the Distributions")

    # Boxplot
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df, ax=ax3)
    ax3.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax3.set_title('Boxplot of Weekly Sales: Holiday vs Non-Holiday')
    st.pyplot(fig3)

    # Histogram
    fig4, ax4 = plt.subplots()
    sns.histplot(holiday_sales, kde=True, color='green', label='Holiday', ax=ax4, stat='density', bins=30)
    sns.histplot(non_holiday_sales, kde=True, color='red', label='Non-Holiday', ax=ax4, stat='density', bins=30)
    ax4.set_title('Sales Distribution: Holiday vs Non-Holiday Weeks')
    ax4.legend()
    st.pyplot(fig4)

    st.write("----")

    st.subheader("Step 4: Levene’s Test for Variance Comparison")

    stat, p = levene(holiday_sales, non_holiday_sales)
    st.write(f"**Levene’s Test statistic** = {stat:.4f}, **p-value** = {p:.4f}")

    if p < 0.05:
        st.success("✅ There is a significant difference in variance between holiday and non-holiday weeks.")
    else:
        st.error("❌ No significant difference in variance. Our hypothesis is not supported.")

    st.write("Levene’s test for equality of variance yielded a test statistic of 2.6775 and a p-value of 0.1018. Since the p-value is greater than the standard significance level of 0.05, we fail to reject the null hypothesis. This suggests that there is no statistically significant difference in the variance of weekly sales between holiday and non-holiday weeks")
    st.write("Our hypothesis that holiday weeks would show significantly higher variance in sales is not supported by the statistical test.")
    st.write("----")

