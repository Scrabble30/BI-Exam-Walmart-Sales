import streamlit as st
import descriptive_stats
import linearregression
import clustering
import classification


def show_homepage():
    st.image('Images/Walmart3.png', use_column_width=True)
    st.title('Walmart BI Project')
    st.write('Group 6 - Made by: Bekhan, Otto, Victor & Patrick')

    st.header("""Sprint 1: Problem Formulation""")
    st.write("---------------")
    st.subheader("Brainstorm:")
    st.write("After reviewing publicly available datasets on Kaggle, we selected the Walmart Sales Forecasting dataset due to its real-world relevance and variety of features such as temperature, fuel prices, CPI, unemployment, and holiday indicators. This dataset is well-suited for exploring predictive modeling, segmentation, and descriptive analytics in a retail setting using Business Intelligence (BI) and Artificial Intelligence (AI) techniques. It opens up opportunities to enhance sales planning, resource allocation, and marketing strategies.")
    st.write("---------------")

    st.subheader("Problem Statement & Annotation:")
    st.write("Challenge:")
    st.write("We aim to address the challenge of predicting and understanding fluctuations in weekly retail sales across Walmart stores based on economic and environmental variables.")
    st.write("")
    st.write("Why this is important:")
    st.write("ccurate sales forecasting is essential for inventory control, staffing, and marketing decisions. With rising operational costs and shifting consumer behavior, data-driven insights are more critical than ever in retail.")
    st.write("")
    st.write("Expected Solution:")
    st.write("We will apply regression, clustering, classification, and descriptive statistical methods to analyze and model weekly sales performance. This includes identifying patterns, forecasting future sales, and segmenting store behavior profiles.")
    st.write("")
    st.write("Positive Impact:")
    st.write("The results will enable store managers, analysts, and corporate strategists to make more informed decisions, reduce waste, improve promotions, and optimize customer satisfaction.")
    st.write("---------------")

    st.subheader("Research Questions:")
    st.markdown("""
    <div style="border: 0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ1: Regression</strong></li>
            <li><strong>Question:</strong> How do temperature, fuel prices, and unemployment affect weekly sales performance?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>H:</strong> Weekly sales can be predicted using a multilinear regression model with temperature, fuel price, and unemployment as predictors.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div style="border: 0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ2: Clustering</strong></li>
            <li><strong>Question:</strong> What natural groupings of store sales patterns can be identified?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>H:</strong> Clustering stores based on average weekly sales and economic context (fuel price, CPI, unemployment) will reveal distinct store behavior profiles (e.g. "holiday-sensitive stores", "price-sensitive stores").</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div style="border: 0px solid #ccc; color: white; padding: 15px; border-radius: 8px; background-color: #262730;">
        <ul style="margin-top: 0; list-style-type: none; padding-left: 0;">
            <li><strong>RQ3: Classification</strong></li>
            <li><strong>Question:</strong> Can we classify whether a store’s weekly sales will be above or below average?</li>
            <li>______________________________________________________________________________</li>
            <li><strong>H:</strong> Weekly sales can be reliably categorized as “high” or “low” using classification models with predictors like fuel price, CPI, unemployment, and holiday flags.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
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
    
    st.write("")

    st.subheader("🛠 Tools & Platforms")
    st.markdown("""
        - Version Control: GitHub repository for code and documentation
        - Development: Jupyter Notebooks / VS Code
        - Libraries: pandas, matplotlib, scikit-learn, statsmodels, streamlit
        - BI Platform: Power BI or Tableau for interactive dashboards
    """)
    st.write("")

    st.subheader("Data Sources & Links")
    st.markdown("""      
    - https://www.kaggle.com/datasets/yasserh/walmart-dataset
    """)





def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home' 
    st.sidebar.image('Images/Walmart.jpg', caption='Wallmart Store in USA', use_column_width=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🏠 Home Page"):
        st.session_state.page = 'home'
    if st.sidebar.button("Data Wrangling"):
        st.session_state.page = 'wrangling'
    if st.sidebar.button("RQ 1 - Linear Regression"):
        st.session_state.page = 'linearregression'
    if st.sidebar.button("RQ 2 - Clustering"):
        st.session_state.page = 'clustering'
    if st.sidebar.button("RQ 3 - Classification"):
        st.session_state.page = 'classification'
    if st.sidebar.button("RQ 4 - Descriptive Statistics"):
        st.session_state.page = "statistics"

    # Conditional display
    if st.session_state.page == 'home':
        show_homepage()
    elif st.session_state.page == 'wrangling':
        st.write("Data Wrangling Page - To be implemented")
    elif st.session_state.page == 'linearregression':
        linearregression.regression()
    elif st.session_state.page == 'clustering':
        clustering.clustering()
    elif st.session_state.page == 'classification':
        classification.classification()
    elif st.session_state.page == 'statistics':
        descriptive_stats.descriptivestatistics()


if __name__ == "__main__":
    main()
