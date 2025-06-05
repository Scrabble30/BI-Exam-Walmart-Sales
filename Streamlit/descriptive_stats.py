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
    st.write("Here we can get our answer from two columns, Weekly_Sales and Holiday_Flag")

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

    st.write("It turns out that 6.9% of the data has holiday weeks in them ")    

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

    st.write("Holiday weeks generally have higher average sales than non-holiday weeks, as expected due to increased consumer spending. The median is also higher, suggesting that not only are the extreme weeks stronger, but the central tendency of the data shifts upward.")

    st.write("----")

    st.subheader("Step 3: Visualizing the Distributions")

    # Boxplot
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df, ax=ax3)
    ax3.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax3.set_title('Boxplot of Weekly Sales: Holiday vs Non-Holiday')
    st.pyplot(fig3)

    st.write("The rate of sales on holidays is higher than on other days. Which can be explained by the KDE curves under here")

    # Histogram
    fig4, ax4 = plt.subplots()
    sns.histplot(holiday_sales, kde=True, color='green', label='Holiday', ax=ax4, stat='density', bins=30)
    sns.histplot(non_holiday_sales, kde=True, color='red', label='Non-Holiday', ax=ax4, stat='density', bins=30)
    ax4.set_title('Sales Distribution: Holiday vs Non-Holiday Weeks')
    ax4.legend()
    st.pyplot(fig4)

    st.write("x-axis is weekly sales, in millions")
    st.write("y-axis is density")

    st.write("**Red curve (Non-Holiday)**")
    st.write("The non-holiday curve peaks higher than the green one — at around 500,000 - 600,000 in weekly sales.")
    st.write("This tells us that non-holiday weeks consistently fall around lower sales values, which aligns with the lower mean and median.")

    st.write("**Green curve (Holiday)**")
    st.write("The holiday distribution is more spread out, with a flatter curve extending further into higher sales.")
    st.write("The green bars are more prominent in the right tail (e.g., above 1.5M), where holiday weeks dominate.")

    st.write("**Overlap area:**")
    st.write('There is significant overlap between the two distributions in the 500K - 1.5M range, but holiday weeks extend further into the high-sales range.')
    st.write("That is consistent with the higher max, mean, and std dev for holiday weeks.")

    st.write("----")

    st.subheader("Step 4: Levene’s Test for Variance Comparison")

    stat, p = levene(holiday_sales, non_holiday_sales)
    st.write(f"**Levene’s Test statistic** = {stat:.4f}, **p-value** = {p:.4f}")

    if p < 0.05:
        st.success("✅ There is a significant difference in variance between holiday and non-holiday weeks.")
    else:
        st.error("❌ No significant difference in variance. Our hypothesis is not supported.")

    st.write("Levene’s test for equality of variance yielded a test statistic of 2.6775 and a p-value of 0.1018. Since the p-value is greater than the standard significance level of 0.05, we fail to reject the null hypothesis. This suggests that there is no statistically significant difference in the variance of weekly sales between holiday and non-holiday weeks")
    st.subheader("Conclusion:")
    st.write("Our hypothesis that holiday weeks would show significantly higher variance in sales is not supported by the statistical test.")
    st.markdown("""
    There are several possible reasons why we did not find a significant difference in variance:

    - **Sample size and holiday data proportion:** Only about 6.9% of weeks are holiday weeks, which might limit the statistical power to detect variance differences.
    - **Seasonality effects:** Seasonal trends may overshadow holiday effects, causing holiday variance to appear less distinct.
    - **Data quality and cleaning:** Removing outliers improves precision but might also remove extreme holiday spikes that contribute to variance.
    - **Other factors:** External influences like promotions, economic conditions, or regional differences could affect sales variability beyond holidays.

    These considerations highlight the complexity of sales data and suggest that further investigation, possibly with more granular data or additional features, may be needed to fully understand variance drivers.
    """)
    st.write("----")

    st.subheader("Step 5: Investigating Seasonality to Explain Variance")

    st.write("""
        Since our test showed **no significant difference in variance** between holiday and non-holiday weeks, 
        we explore whether **seasonal patterns** may better explain variation in weekly sales.
    """)

    # Derive season from date
    def season_getter(quarter):
        return {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}.get(quarter, 'Unknown')

    df['Quarter'] = df['Date'].dt.quarter
    df['Season'] = df['Quarter'].apply(season_getter)

    # Total sales by season
    season_sales = df.groupby('Season')['Weekly_Sales'].agg(['sum', 'mean', 'std']).reindex(['Winter', 'Spring', 'Summer', 'Fall'])

    # Plot total sales
    st.markdown("### Total Weekly Sales by Season")
    fig5, ax5 = plt.subplots()
    sns.barplot(x=season_sales.index, y=season_sales['sum'], palette='viridis', ax=ax5)
    ax5.set_ylabel("Total Sales")
    ax5.set_title("Total Weekly Sales by Season")
    st.pyplot(fig5)

    # Plot average weekly sales
    st.markdown("### Average Weekly Sales by Season")
    fig6, ax6 = plt.subplots()
    sns.barplot(x=season_sales.index, y=season_sales['mean'], palette='coolwarm', ax=ax6)
    ax6.set_ylabel("Average Weekly Sales")
    ax6.set_title("Average Weekly Sales by Season")
    st.pyplot(fig6)

    # Optional: std deviation per season
    st.markdown("### Standard Deviation of Weekly Sales by Season")
    fig7, ax7 = plt.subplots()
    sns.barplot(x=season_sales.index, y=season_sales['std'], palette='magma', ax=ax7)
    ax7.set_ylabel("Standard Deviation")
    ax7.set_title("Sales Variability per Season")
    st.pyplot(fig7)

    # Interpretation
    st.markdown("""
        ### Interpretation
        
        The analysis shows that **seasonal trends** have a noticeable effect on sales levels and variability.
        
        - **Summer and Fall** tend to have higher total and average sales.
        - **Standard deviation is also larger** in some seasons than others.

        This suggests that **natural seasonality** across the year may account for more variation 
        in sales than whether or not a week contains a holiday.

        Thus, the lack of significant difference in holiday variance (from Levene’s Test) might be because
        **seasonality is a stronger underlying driver**.
    """)

    