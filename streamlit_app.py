#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantipy GUI - Streamlit Application
Main entry point for the Quantipy data analysis interface
"""

import streamlit as st
import sys
import os

# Configure page
st.set_page_config(
    page_title="Quantipy GUI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">📊 Quantipy GUI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Python for People Data - Interactive Analysis Interface</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
## Welcome to Quantipy GUI

This interactive application provides a user-friendly interface for **Quantipy**,
a powerful data processing and analysis tool designed for survey and market research data.

### Features:
- 📁 **Data Import**: Load data from multiple formats (Quantipy, SPSS, CSV)
- 🔍 **Data Exploration**: Browse variables, view metadata, and crosstabs
- 📊 **Analysis**: Create batches, define cross-tabulations, and generate statistics
- 📈 **Visualization**: View results with interactive charts and tables
- 💾 **Export**: Save results to Excel, SPSS, or other formats

### Getting Started:

Use the sidebar navigation to access different features:

1. **📁 Data Loader** - Import and manage your datasets
2. **🔍 Data Explorer** - Explore variables and metadata
3. **📊 Analysis** - Create batches and run analyses
4. **📈 Results** - View and export results

---

### Quick Start Example:

""")

# Show example workflow
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Step 1: Load Data
    - Upload CSV/JSON files
    - Or use example data
    - Select SPSS or other formats
    """)

with col2:
    st.markdown("""
    #### Step 2: Explore
    - Browse variables
    - View crosstabs
    - Check metadata
    """)

with col3:
    st.markdown("""
    #### Step 3: Analyze
    - Create batches
    - Define x/y variables
    - Generate results
    """)

st.markdown("---")

# Sample data section
st.subheader("📦 Load Example Data")
st.markdown("""
Click below to load the example dataset and start exploring immediately:
""")

if st.button("🚀 Load Example Data & Explore", type="primary"):
    st.session_state.load_example = True
    st.switch_page("pages/01_Data_Loader.py")

st.markdown("---")

# Information section
with st.expander("ℹ️ About Quantipy"):
    st.markdown("""
    **Quantipy** is an open-source data processing, analysis and reporting software project
    that builds on the excellent pandas and numpy libraries. Aimed at people data, Quantipy
    offers support for:

    - Native handling of special data types like multiple choice variables
    - Statistical analysis using case or observation weights
    - DataFrame metadata management
    - Pretty data exports to multiple formats

    **Key Features:**
    - Multiple data format support (CSV, SPSS, Dimensions, Decipher, Ascribe)
    - Open metadata format to describe and manage datasets
    - Powerful, metadata-driven data cleaning and transformation
    - Computation and assessment of data weights
    - Easy-to-use analysis interface with Batch definitions
    - Structured reporting via Chain and Cluster containers
    - Export to SPSS, Excel, PowerPoint with flexible layouts

    **Documentation:** [readthedocs.org/quantipy](http://quantipy.readthedocs.io/)
    """)

with st.expander("💡 Tips for Using This App"):
    st.markdown("""
    - **Navigation**: Use the sidebar to switch between different pages
    - **Data Persistence**: Your loaded dataset stays in memory across pages
    - **Session State**: The app maintains your work during the session
    - **File Upload**: Supports drag-and-drop for easy file loading
    - **Export**: Generate reports in multiple formats from the Results page
    - **Example Data**: Use the built-in example dataset to explore features
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <small>Quantipy GUI | Built with Streamlit | Python 2.7 Compatible</small>
</div>
""", unsafe_allow_html=True)
