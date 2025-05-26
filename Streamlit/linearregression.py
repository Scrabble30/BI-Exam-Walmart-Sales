import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score


def regression():
    st.title("RQ 1 - Linear Regression")

    st.markdown("""
    <div style="border: 0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ1: Regression</strong></li>
            <li><strong>Question:</strong> How do temperature, fuel prices, and unemployment affect weekly sales performance?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>Hypothesis:</strong> Weekly sales can be predicted using a multilinear regression model with temperature, fuel price, and unemployment as predictors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.header("Linear Regression Explanation")
    st.markdown("""
    Linear regression models the relationship between one dependent variable and multiple independent variables.
    It assumes this relationship is linear and can help make predictions or understand how variables influence each other.
    
    This model fits a straight line through the data to minimize prediction error.
    """)

    st.subheader("When to Use Linear Regression")
    st.markdown("""
    - The relationship between variables is approximately linear.
    - Predicting a continuous numeric value.
    - Independent variables are not highly correlated (low multicollinearity).
    - Errors are normally distributed with constant variance.
    - When interpretability is important.
    """)

    st.header("Data Loading")
    df = pd.read_csv("../Data/Cleaned-Walmart_Sales.csv")
    st.write("Initial Dataset Info:")
    st.dataframe(df.head())

    df = df.drop(columns=['Date', 'Store'])
    df = df.astype({col: 'float' for col in df.select_dtypes(include=['int', 'float']).columns})

    st.markdown("### Cleaned Dataset")
    st.write(df.info())
    st.write(df.describe())

    st.markdown("### Missing Values Check")
    st.write(df.isnull().sum())

    st.header("Outlier Removal")
    rows_before = len(df)
    cols_to_clean = ['Weekly_Sales']
    for col in cols_to_clean:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    rows_after = len(df)
    st.write(f"Rows before: {rows_before}")
    st.write(f"Rows after:  {rows_after}")
    st.write(f"Total rows removed: {rows_before - rows_after}")

    st.header("Data Visualizations")

    st.subheader("Pairplot")
    st.markdown("""
    Visualizing relationships between features and the target variable.
    """)
    st.pyplot(sns.pairplot(df, x_vars=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'], 
                           y_vars='Weekly_Sales', kind='scatter'))

    st.subheader("Histograms")
    st.markdown("""
    Histograms help us understand the distribution of each numeric feature.
    """)
    fig, ax = plt.subplots(figsize=(10, 8))
    df.hist(bins=20, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Boxplots")
    st.markdown("""
    Boxplots show the spread and detect outliers. Each feature's range and central tendency is visualized.
    """)
    numeric_cols = df.select_dtypes(include=['number']).columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, column in enumerate(numeric_cols):
        axes[i].boxplot(df[column].dropna())
        axes[i].set_title(column)
        axes[i].set_ylabel(column)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Boxplots of Numeric Columns', fontsize=16)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    st.markdown("""
    Shows the strength and direction of linear relationships.
    """)
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.header("Model Building")
    st.markdown("""
    We will train a linear regression model using temperature, fuel price, and unemployment.
    """)
    feature_cols = ['Temperature', 'Fuel_Price', 'Unemployment']
    X = df[feature_cols]
    y = df['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    st.subheader("Model Coefficients")
    st.write(f"Intercept (b0): {linreg.intercept_}")
    st.write("Feature Coefficients:")
    st.write(dict(zip(feature_cols, linreg.coef_)))

    st.markdown("""
    - Coefficients represent how much weekly sales change with one unit increase in each variable.
    - Negative for temperature suggests that higher temperatures slightly reduce weekly sales.
    - Positive for fuel price and unemployment (if observed) could be counterintuitive.
    """)

    st.subheader("Model Prediction and Evaluation")
    y_predicted = linreg.predict(X_test)

    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R2 Score: {r2:.4f}")

    st.subheader("Actual vs Predicted Plot")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted, color='orange', alpha=0.6)
    ax.plot(y_test, y_test, color='blue', label='Actual')
    ax.set_xlabel("Actual Weekly Sales")
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title("Actual vs Predicted Weekly Sales")
    ax.legend()
    st.pyplot(fig)

    st.header("Conclusion")
    st.markdown("""
    - The correlation between features and weekly sales is weak.
    - The model does not perform well (low R² score).
    - While unemployment had a slightly intuitive effect, fuel price and temperature didn’t provide useful predictive power.
    
    This suggests that a simple linear model is insufficient and we may need to explore additional variables or more complex models.
    """)