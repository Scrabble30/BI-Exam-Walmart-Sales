import streamlit as st
import pandas as pd

def clustering():
    st.title("RQ 2 - Clustering")

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