import streamlit as st
import pandas as pd

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

    st.subheader("How we find out:")
    st.write("We have to remember that we are working out from the cleaned Walmart Sales. In here we have removed some outliers to get some more precise data")
