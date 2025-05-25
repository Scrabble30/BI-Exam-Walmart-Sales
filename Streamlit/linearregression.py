import streamlit as st
import pandas as pd

def regression():
    st.title("RQ 1 - Linear Regression")
    
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