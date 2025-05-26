# classification.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    roc_curve, auc
)
import joblib

def classification():
    st.title("RQ 3 - Classification")

    st.markdown("""
    <div style="border: 0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ3: Classification</strong></li>
            <li><strong>Can we classify whether a given week’s sales (across all stores) will be above or below the average, using only general predictors like fuel price, CPI, unemployment, and holiday flags?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>H: </strong>Weekly sales can be reliably categorized as “high” or “low” using classification models with predictors like fuel price, CPI, unemployment, and holiday flags.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Introduction

    ### Objective

    This notebook aims to determine whether general economic and temporal factors can predict if weekly sales are above or below average across Walmart stores. Store-specific identifiers are excluded to avoid models that memorize individual store patterns, focusing instead on generalizable insights.

    ### Approach

    Exploratory analysis in the DataWrangling notebook showed very little linear correlation between most features and weekly sales, suggesting that linear models may struggle to capture meaningful relationships. However, classification models—particularly tree-based methods—are well-suited to detect nonlinear patterns, motivating their use here.

    To improve model performance, features such as year, month, and week are extracted from the date column. Scaling techniques—including StandardScaler for normally distributed features and MinMaxScaler for others—are applied to normalize the data. Several classification algorithms are trained and evaluated, including Decision Tree, Random Forest, and Gaussian Naive Bayes, to identify the most effective approach.

    ### Note

    While classification is not typically used for continuous targets like sales, reframing the problem as a binary classification simplifies interpretation and allows assessment of the predictive power of general features without relying on store-specific data.
    """)

    # --- Data Ingestion ---
    st.markdown("""
    ## Data Ingestion

    In this section, we load the Walmart Sales dataset and perform initial exploration to understand its structure and contents. We check for missing values, duplicates, and get a summary of the data to ensure it is ready for further processing.
    """)

    data_path = st.text_input("Enter path to Walmart Sales CSV:", "../Data/Walmart_Sales.csv")
    if not data_path:
        st.warning("Please provide a valid CSV path to proceed.")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    st.write("**Columns:**", list(df.columns))
    st.write("**Shape:**", df.shape)
    st.write("**Summary statistics:**")
    st.dataframe(df.describe())
    st.write("**Missing values per column:**")
    st.write(df.isnull().sum())
    st.write("**Number of duplicate rows:**", df.duplicated().sum())
    st.write("**Sample rows:**")
    st.dataframe(df.sample(5))

    # --- Data Cleaning ---
    st.markdown("""
    ## Data Cleaning

    In this section, we prepare the dataset for analysis by removing unnecessary columns, converting data types, and addressing potential outliers. These steps help ensure data quality and improve the reliability of later modeling.
    """)

    st.write(
        "We remove the 'Store' column and convert the 'Date' column from text to datetime. "
        "This prepares the data for further processing and feature extraction."
    )

    if 'Store' in df.columns:
        df.drop(['Store'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    st.markdown("""
    ### Outlier Removal
    """)
    # Boxplot before outlier removal
    st.write("**Boxplot before outlier removal:**")

    # List of columns to plot
    cols = ['Weekly_Sales', 'Temperature', 'Unemployment']

    # Create subplots: one boxplot per variable, arranged horizontally
    fig, axes = plt.subplots(1, len(cols), figsize=(10, 5), sharey=False)

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], vert=True, patch_artist=True)
        axes[i].set_title(col, fontsize=14)
        axes[i].grid(True, axis='y')

    plt.tight_layout()
    st.pyplot(fig)

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        num_outliers = outliers.shape[0]

        st.write(f"Removed {num_outliers} outliers from '{column}'.")

        # Return DataFrame with outliers removed
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    for col in ['Weekly_Sales', 'Temperature', 'Unemployment']:
        df = remove_outliers(df, col)

    st.write("**Boxplot after outlier removal:**")
    # List of columns to plot
    cols = ['Weekly_Sales', 'Temperature', 'Unemployment']

    # Create subplots: one boxplot per variable, arranged horizontally
    fig, axes = plt.subplots(1, len(cols), figsize=(10, 5), sharey=False)

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col], vert=True, patch_artist=True)
        axes[i].set_title(col, fontsize=14)
        axes[i].grid(True, axis='y')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Feature Engineering ---
    st.markdown("""
    ## Feature Engineering

    In this section, we create new features from existing data to improve model performance. This includes extracting temporal features from the date column and generating a binary target variable for classification. Scaling is handled in the next section.

    **Note:** We also experimented with extracting a "Season" feature from the date, but it did not improve model performance and had zero feature importance, so it was removed from the final feature set.
    """)

    st.write(
    "To prepare our data for modeling, we extract new columns for year, month, quarter, and week from the original date information. "
    "We also create a new 'High_Sales' column, which marks weeks with above-average sales. "
    "Finally, we remove the original 'Date' and 'Weekly_Sales' columns, since their information is now captured in the new features."
    )

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Week'] = df['Date'].dt.isocalendar().week
    df.drop(['Date'], axis=1, inplace=True)
    mean_sales = df['Weekly_Sales'].mean()
    df['High_Sales'] = (df['Weekly_Sales'] > mean_sales).astype(int)
    df.drop(['Weekly_Sales'], axis=1, inplace=True)

    st.write("**Engineered features preview:**")
    st.dataframe(df.head())

    # --- Feature Selection & Scaling ---
    st.markdown("""
    ## Feature Selection & Scaling

    In this section, we analyze feature relationships and prepare the data for modeling. We first examine the correlation matrix to check for strong linear relationships or redundancy among features. Next, we use histograms to visually assess the distribution of each feature, which guides our choice of scaling method.
    """)

    corr = df.corr()
    st.markdown("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    As observed, most features show very little linear correlation with the target, so no features are removed at this stage.

    **Note on Unemployment and Sales Correlation**

    The correlation matrix shows a very weak negative correlation (-0.07) between unemployment and high sales weeks. This suggests a slight tendency for higher unemployment to be associated with lower sales, but the relationship is too weak to be meaningful. While this aligns with the intuition that higher unemployment might reduce consumer spending, the data does not provide strong evidence for this effect.
    """)

    feature_columns = [col for col in df.columns if col != 'High_Sales']
    st.write("**Histograms of features:**")
    fig, ax = plt.subplots(figsize=(10, 8))
    df[feature_columns].hist(bins=20, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    Based on the histograms, StandardScaler is applied to features that are roughly normally distributed, while MinMaxScaler is used for features that are not. This ensures that all features are on comparable scales.
    """)

    features_norm = ['Temperature', 'Unemployment']
    features_non_norm = [col for col in df.columns if col not in features_norm + ['High_Sales']]

    st.write("### Feature Scaling Overview")
    st.write(
        "We scale each group of features differently based on their distribution:"
    )

    st.markdown("**Features scaled with StandardScaler (normally distributed):**")
    st.markdown('\n'.join([f"- {col}" for col in features_norm]))

    st.markdown("**Features scaled with MinMaxScaler (not normally distributed):**")
    st.markdown('\n'.join([f"- {col}" for col in features_non_norm]))

    standardScaler = StandardScaler()
    X_norm = pd.DataFrame(standardScaler.fit_transform(df[features_norm]), columns=features_norm)
    scaler = MinMaxScaler()
    X_non_norm = pd.DataFrame(scaler.fit_transform(df[features_non_norm]), columns=features_non_norm)
    X = pd.concat([X_non_norm, X_norm], axis=1)[feature_columns]
    y = df['High_Sales']

    st.write("**Scaled feature preview:**")
    st.dataframe(X.head())

    st.write("**Histograms after scaling:**")
    fig, ax = plt.subplots(figsize=(10, 8))
    X.hist(bins=20, ax=ax)
    st.pyplot(fig)

    # --- Model Training & Evaluation ---
    st.markdown("""
    ## Model Training & Evaluation
    
    In this section, we train and evaluate several classification models to determine which performs best at predicting whether weekly sales are above or below average. We use Decision Tree, Random Forest, and Gaussian Naive Bayes classifiers. Model performance is assessed using accuracy, classification reports, confusion matrices, ROC curves, and cross-validation.
    """)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("**Train/Test split shapes:**")
    st.write("X_train:", X_train.shape, "X_test:", X_test.shape)
    st.write("y_train:", y_train.shape, "y_test:", y_test.shape)

    st.markdown("""
    We evaluated three classification models on the task of predicting whether a week’s sales are above or below average:

    - **Decision Tree**
    - **Random Forest**
    - **Gaussian Naive Bayes**

    Below you’ll find the main performance metrics, confusion matrices, and ROC curves for each model, along with a summary and interpretation of the results.
    """)

    model_choice = st.selectbox(
        "Choose a classification model to train and evaluate:",
        ("Decision Tree", "Random Forest", "Gaussian Naive Bayes")
    )

    # Load a model
    def load_model(filename):    
        try:
            return joblib.load(filename)
        except Exception as e:
            st.error(f"Could not load data: {e}")
            return

    def show_classification_report(model):
        y_pred = model.predict(X_test)

        st.markdown("#### Accuracy & Classification Report")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2%}")
        st.text(classification_report(y_test, y_pred))

    def show_confusion_matrix(model):
        y_pred = model.predict(X_test)

        st.markdown("#### Confusion Matrix")
        st.write(
            "The confusion matrix summarizes the model’s classification performance by showing the counts of true positives, false positives, true negatives, and false negatives."
        )

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='.0f', ax=ax, cmap="Blues")
        plt.xlabel("Predicted High Sales"); plt.ylabel("True High Sales")
        plt.title(f"{model_choice} Confusion Matrix")
        st.pyplot(fig)

    def show_roc_curve(model):
        if hasattr(model, "predict_proba"):
            st.markdown("""
                #### ROC Curve
                        
                The ROC curve shows how well the model separates the two classes across different thresholds. The AUC score summarizes this performance; higher values mean better distinction between high and low sales weeks.
            """)
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'{model_choice} ROC Curve')
            plt.legend()
            st.pyplot(fig)
        else:
            st.info("ROC curve not available for this model.")

    def show_cross_validation(model):
        st.markdown("""
            #### Cross-Validation
            
            To assess the robustness of the Decision Tree model, we perform 5-fold cross-validation on the entire dataset. This provides an estimate of how well the model generalizes to unseen data.
        """)
        cv_scores = cross_val_score(model, X, y, cv=5)
        st.write("Cross-validation scores:", cv_scores)
        st.write("Mean CV accuracy:", cv_scores.mean())

    def show_feature_importance(model):
        if model_choice in ["Decision Tree", "Random Forest"]:
            st.markdown("""
                #### Feature Importance

                The Decision Tree model provides feature importances, indicating which predictors most influence its classification decisions.
            """)
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            st.dataframe(feature_importance_df)
        else:
            st.info("Feature importance is not available for Gaussian Naive Bayes.")

    def show_train_vs_test_accuracy(model):
        st.markdown("""
            #### Train vs. Test Accuracy

            Comparing train and test accuracy helps assess whether the model is overfitting or underfitting.
        """)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        st.write(f"**Train Accuracy:** {train_acc:.2f}")
        st.write(f"**Test Accuracy:** {test_acc:.2f}")

    if model_choice == "Decision Tree":
        st.subheader("DecisionTreeClassifier")
        st.write(
            "We first train a Decision Tree classifier. Through experimentation, a maximum depth of 10–12 was found to yield the best results. We evaluate the model using accuracy, classification report, confusion matrix, ROC curve, cross-validation, feature importance, and train/test accuracy."
        )

        decisionTreeClassifier = load_model('../Data/DecisionTreeClassifier.pkl')

        show_classification_report(decisionTreeClassifier)
        st.markdown("""
            The Decision Tree model achieves an overall accuracy of 66%. Precision is higher for low sales weeks (class 0, precision: 0.72) than for high sales weeks (class 1, precision: 0.60). Recall is slightly higher for high sales weeks (0.67) than for low sales weeks (0.65). The F1-scores are 0.68 for low sales and 0.63 for high sales, indicating reasonably balanced performance. Overall, the model is somewhat better at correctly identifying low sales weeks, but still captures a substantial portion of high sales weeks as well.
        """)

        show_confusion_matrix(decisionTreeClassifier)
        st.markdown("""
            The confusion matrix summarizes how well the classifier distinguishes between high and low sales weeks:

            - **True Negatives (TN):** 432 weeks correctly predicted as low sales  
            - **False Positives (FP):** 234 weeks incorrectly predicted as high sales  
            - **False Negatives (FN):** 170 weeks incorrectly predicted as low sales  
            - **True Positives (TP):** 348 weeks correctly predicted as high sales

            This means the model is better at identifying weeks with low sales than weeks with high sales. The relatively high number of false positives and false negatives suggests there is still room for improving the model’s precision and recall for both classes.
        """)

        show_roc_curve(decisionTreeClassifier)
        st.markdown("""
            The ROC curve visualizes the trade-off between the true positive rate (sensitivity) and the false positive rate.  
            An area under the curve (AUC) of 0.74 indicates that the model has a reasonable ability to distinguish between high and low sales weeks, but is not highly accurate. A perfect model would have an AUC of 1.0, while a value of 0.5 would indicate random guessing.
        """)

        show_cross_validation(decisionTreeClassifier)
        st.markdown("""
            The cross-validation scores vary considerably across folds, ranging from about 15% to 33%, with a mean accuracy of 23.21%. This suggests that the model's performance is unstable and may not generalize well to new data, possibly due to class imbalance, data splits, or model limitations.
        """)

        show_feature_importance(decisionTreeClassifier)
        st.markdown("""
            Unemployment is the most important feature for the Decision Tree, followed by CPI and Temperature. Features like Month, Quarter, and Holiday_Flag contribute very little to the model's decisions.
        """)

        show_train_vs_test_accuracy(decisionTreeClassifier)
        st.markdown("""
            The model achieves 74% accuracy on the training set and 66% accuracy on the test set. The moderate gap suggests some overfitting, but the model still generalizes reasonably well to unseen data.
                """)
    elif model_choice == "Random Forest":
        st.subheader("Random Forest Classifier")

        randomForestClassifier = load_model('../Data/RandomForestClassifier.pkl')

        show_classification_report(randomForestClassifier)
        st.markdown("""
            The Random Forest model achieves 66% accuracy. It performs slightly better at identifying low sales weeks (precision 0.69, recall 0.71) than high sales weeks (precision 0.61, recall 0.60), with balanced F1-scores for both classes.
        """)

        show_confusion_matrix(randomForestClassifier)
        st.markdown("""
            - True Negatives (TN): 470 weeks correctly predicted as low sales  
            - False Positives (FP): 196 weeks incorrectly predicted as high sales  
            - False Negatives (FN): 209 weeks incorrectly predicted as low sales  
            - True Positives (TP): 309 weeks correctly predicted as high sales

            The model is slightly better at identifying low sales weeks, but still captures a fair number of high sales weeks.
        """)

        show_roc_curve(randomForestClassifier)
        st.markdown("""
            The Random Forest model achieves an AUC of 0.74, indicating a reasonable ability to distinguish between high and low sales weeks.
        """)

        show_cross_validation(randomForestClassifier)
        st.markdown("""
            Cross-validation scores range from about 15% to 43%, with a mean accuracy of 27.58%. This variability suggests that model performance is unstable across different data splits.
        """)

        show_feature_importance(randomForestClassifier)
        st.markdown("""
            Unemployment is the most important feature, followed by CPI and Temperature. Holiday_Flag and Quarter contribute very little to the model’s predictions.            
        """)

        show_train_vs_test_accuracy(randomForestClassifier)
        st.markdown("""
            The model achieves 74% accuracy on the training set and 66% on the test set, suggesting some overfitting but reasonable generalization to new data.
        """)
    else:
        st.subheader("Gaussian Naive Bayes Classifier")

        bayes = load_model('../Data/GaussianNB.pkl')

        show_classification_report(bayes)
        st.markdown("""
            The Gaussian Naive Bayes model achieves 55% accuracy. It is better at identifying low sales weeks (precision 0.58, recall 0.77) than high sales weeks (precision 0.48, recall 0.27). The low recall and F1-score for high sales weeks indicate the model struggles to correctly identify those cases.
        """)

        show_confusion_matrix(bayes)
        st.markdown("""
            - True Negatives (TN): 516 weeks correctly predicted as low sales  
            - False Positives (FP): 150 weeks incorrectly predicted as high sales  
            - False Negatives (FN): 380 weeks incorrectly predicted as low sales  
            - True Positives (TP): 138 weeks correctly predicted as high sales

            The model is much more likely to correctly identify low sales weeks than high sales weeks, as seen by the high number of false negatives for the high sales class.
        """)

        show_roc_curve(bayes)
        st.markdown("""
            The model’s ROC AUC is 0.57, only slightly better than random guessing, indicating poor ability to distinguish between high and low sales weeks.
        """)

        show_cross_validation(bayes)
        st.markdown("""
            Cross-validation scores range from about 41% to 55%, with a mean accuracy of 49.65%. This further suggests the model does not generalize well and is not effective for this classification task.
        """)

        st.markdown("""
            #### Feature Importance

            Feature importance is not available for Gaussian Naive Bayes, as this model does not provide a direct measure of how much each feature contributes to predictions. Unlike tree-based models, Naive Bayes assumes all features contribute independently and equally based on their statistical distribution.
        """)

        show_train_vs_test_accuracy(bayes)
        st.markdown("""
            The model achieves 56% accuracy on the training set and 55% on the test set, indicating that it is not overfitting but also does not perform well on either set.
        """)

    # --- Results & Discussion ---
    st.header("Results & Discussion")
    st.markdown("""
    ### Model Performance Overview

    Three classification models—Decision Tree, Random Forest, and Gaussian Naive Bayes—were trained to predict whether a given week’s sales would be above or below average using general predictors (fuel price, CPI, unemployment, holiday flags, and temporal features). Performance was evaluated using accuracy, precision, recall, F1-score, ROC AUC, and cross-validation, providing a comprehensive view of each model’s strengths and weaknesses.

    #### Model Comparison Table

    | Model                  | Test Accuracy | ROC AUC | Precision (High Sales) | Recall (High Sales) | F1 (High Sales) | Mean CV Accuracy |
    |------------------------|--------------|---------|-----------------------|---------------------|-----------------|------------------|
    | Decision Tree          | 0.66         | 0.74    | 0.60                  | 0.67                | 0.63            | 0.23             |
    | Random Forest          | 0.66         | 0.74    | 0.61                  | 0.60                | 0.60            | 0.28             |
    | Gaussian Naive Bayes   | 0.55         | 0.57    | 0.48                  | 0.27                | 0.34            | 0.50             |

    #### Interpretation of Results

    - **Tree-based models (Decision Tree and Random Forest) outperformed Gaussian Naive Bayes** across almost all metrics. Both achieved 66% test accuracy and ROC AUC of 0.74, indicating a moderate ability to distinguish between high and low sales weeks. Their precision and recall scores for the high sales class were balanced, though slightly favoring the low sales class.  
    - **Gaussian Naive Bayes performed notably worse**, with only 55% accuracy and an ROC AUC of 0.57, barely above random guessing. It struggled to identify high sales weeks (recall 0.27), indicating that the assumption of feature independence does not fit this dataset well.
    - **Cross-validation scores were low and variable for all models**, especially for tree-based models, suggesting that model performance is unstable and may not generalize well to new data. This could be due to class imbalance, limited predictive power in the selected features, or data splits.
    - **Feature importance analysis (for tree-based models) consistently highlighted unemployment, CPI, and temperature as the most influential predictors**, while temporal features and holiday flags contributed little. This suggests that economic indicators are more relevant than calendar-based or holiday effects for this prediction task.

    #### Key Insights and Surprising Findings

    - **No model achieved strong predictive power**: Even the best models only moderately exceeded random guessing, and cross-validation revealed instability. This suggests that general economic and temporal features alone are insufficient for reliably predicting above- or below-average sales at the aggregate level.
    - **Imbalanced performance**: All models were better at identifying low sales weeks than high sales weeks, as shown by higher recall and precision for the low sales class. This is a common issue in imbalanced datasets and highlights the importance of using multiple metrics for evaluation.
    - **Feature limitations**: The lack of strong predictors—especially the absence of store-specific or promotional data—likely limits the models’ effectiveness. The most important features (unemployment, CPI, temperature) only partially capture the complexity of sales dynamics.

    #### Summary

    - Tree-based models offered the best performance but still left substantial room for improvement.
    - General economic and temporal features alone do not provide enough predictive power for this classification task.
    - Incorporating additional data—such as store-specific factors or promotional information—and exploring more advanced modeling techniques would likely improve results.
    """)

    # --- Conclusion & Limitations ---
    st.header("Conclusion & Limitations")
    st.markdown("""
    This analysis explored whether general economic and temporal features can predict if weekly Walmart sales are above or below average. Tree-based models (Decision Tree and Random Forest) offered the best performance, achieving around 66% accuracy and moderate ability to distinguish high from low sales weeks. However, all models—including Gaussian Naive Bayes—showed limited predictive power, with low and variable cross-validation scores.

    Feature importance analysis revealed that unemployment, CPI, and temperature were the most influential predictors, while holiday and temporal features had little impact. Overall, the results suggest that general features alone are not sufficient for reliable sales classification.

    Limitations of this study include the exclusion of store-specific and promotional data, potential class imbalance, and the limited scope of features. The weak correlations observed indicate that important drivers of sales may be missing from the dataset.

    Future work should focus on incorporating richer data sources, such as store-level information, promotions, and local events, as well as exploring more advanced modeling techniques to improve predictive performance.
    """)