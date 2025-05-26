import streamlit as st
import pandas as pd

def classification():
    st.title("RQ 3 - Classification")

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