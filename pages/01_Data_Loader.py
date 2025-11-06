#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantipy GUI - Data Loader Page
Load datasets from various formats
"""

import streamlit as st
import pandas as pd
import quantipy as qp
import os
import json
import tempfile

st.set_page_config(page_title="Data Loader", page_icon="📁", layout="wide")

st.title("📁 Data Loader")
st.markdown("Load your dataset from various formats or use the example data.")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None

# Check if we should load example data
if st.session_state.get('load_example', False):
    st.session_state.load_example = False  # Reset flag
    try:
        with st.spinner("Loading example dataset..."):
            test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
            name = 'Example Data (A)'
            dataset = qp.DataSet(name, False)
            dataset.read_quantipy(
                os.path.join(test_path, '{}.json'.format(name)),
                os.path.join(test_path, '{}.csv'.format(name))
            )
            st.session_state.dataset = dataset
            st.session_state.dataset_name = name
            st.success("✅ Example dataset loaded successfully!")
    except Exception as e:
        st.error("Error loading example data: {}".format(str(e)))

# Data source selection
st.subheader("Select Data Source")

data_source = st.radio(
    "Choose how to load your data:",
    ["Example Data", "Quantipy Files (JSON + CSV)", "CSV File Only", "SPSS File"],
    horizontal=True
)

st.markdown("---")

# Example Data
if data_source == "Example Data":
    st.info("📦 Load the built-in example dataset to explore Quantipy features")

    if st.button("Load Example Dataset", type="primary"):
        try:
            with st.spinner("Loading example dataset..."):
                test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
                name = 'Example Data (A)'
                dataset = qp.DataSet(name, False)
                dataset.read_quantipy(
                    os.path.join(test_path, '{}.json'.format(name)),
                    os.path.join(test_path, '{}.csv'.format(name))
                )
                st.session_state.dataset = dataset
                st.session_state.dataset_name = name
                st.success("✅ Example dataset loaded successfully!")
                st.rerun()
        except Exception as e:
            st.error("Error loading example data: {}".format(str(e)))

# Quantipy Files
elif data_source == "Quantipy Files (JSON + CSV)":
    st.info("📄 Upload both JSON metadata and CSV data files")

    col1, col2 = st.columns(2)

    with col1:
        json_file = st.file_uploader("Upload JSON Metadata", type=['json'])

    with col2:
        csv_file = st.file_uploader("Upload CSV Data", type=['csv'])

    dataset_name = st.text_input("Dataset Name", value="My Dataset")

    if st.button("Load Quantipy Dataset", type="primary"):
        if json_file is None or csv_file is None:
            st.error("Please upload both JSON and CSV files")
        else:
            try:
                with st.spinner("Loading dataset..."):
                    # Create temporary files
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                        json_content = json.load(json_file)
                        json.dump(json_content, tmp_json)
                        tmp_json_path = tmp_json.name

                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_csv:
                        tmp_csv.write(csv_file.getvalue())
                        tmp_csv_path = tmp_csv.name

                    # Load dataset
                    dataset = qp.DataSet(dataset_name, False)
                    dataset.read_quantipy(tmp_json_path, tmp_csv_path)

                    # Clean up temp files
                    os.unlink(tmp_json_path)
                    os.unlink(tmp_csv_path)

                    st.session_state.dataset = dataset
                    st.session_state.dataset_name = dataset_name
                    st.success("✅ Dataset loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error("Error loading dataset: {}".format(str(e)))
                import traceback
                st.code(traceback.format_exc())

# CSV Only
elif data_source == "CSV File Only":
    st.info("📊 Upload a CSV file (limited functionality without metadata)")
    st.warning("⚠️ Loading CSV without metadata limits Quantipy features. Consider creating metadata or using full Quantipy format.")

    csv_file = st.file_uploader("Upload CSV File", type=['csv'])
    dataset_name = st.text_input("Dataset Name", value="My CSV Dataset")

    if st.button("Load CSV", type="primary"):
        if csv_file is None:
            st.error("Please upload a CSV file")
        else:
            try:
                with st.spinner("Loading CSV..."):
                    # Read CSV into DataFrame
                    df = pd.read_csv(csv_file)

                    # Create basic dataset with minimal metadata
                    dataset = qp.DataSet(dataset_name, False)
                    dataset._data = df

                    # Create basic metadata structure
                    dataset._meta = {
                        'columns': {},
                        'masks': {},
                        'sets': {'data file': {'items': df.columns.tolist()}},
                        'lib': {'default text': 'en-GB', 'values': {}}
                    }

                    # Infer basic metadata for each column
                    for col in df.columns:
                        dataset._meta['columns'][col] = {
                            'name': col,
                            'type': 'string',
                            'text': {dataset._meta['lib']['default text']: col}
                        }

                    dataset.text_key = dataset._meta['lib']['default text']

                    st.session_state.dataset = dataset
                    st.session_state.dataset_name = dataset_name
                    st.success("✅ CSV loaded successfully!")
                    st.warning("Note: Basic metadata was created. You may want to add variable types and value labels.")
                    st.rerun()
            except Exception as e:
                st.error("Error loading CSV: {}".format(str(e)))
                import traceback
                st.code(traceback.format_exc())

# SPSS File
elif data_source == "SPSS File":
    st.info("📈 Upload an SPSS .sav file")

    sav_file = st.file_uploader("Upload SPSS File", type=['sav'])
    dataset_name = st.text_input("Dataset Name", value="My SPSS Dataset")

    if st.button("Load SPSS File", type="primary"):
        if sav_file is None:
            st.error("Please upload an SPSS file")
        else:
            try:
                with st.spinner("Loading SPSS file..."):
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.sav', delete=False) as tmp_sav:
                        tmp_sav.write(sav_file.getvalue())
                        tmp_sav_path = tmp_sav.name

                    # Load SPSS file
                    dataset = qp.DataSet(dataset_name, False)
                    dataset.read_spss(tmp_sav_path)

                    # Clean up temp file
                    os.unlink(tmp_sav_path)

                    st.session_state.dataset = dataset
                    st.session_state.dataset_name = dataset_name
                    st.success("✅ SPSS file loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error("Error loading SPSS file: {}".format(str(e)))
                import traceback
                st.code(traceback.format_exc())

# Display current dataset info
st.markdown("---")
st.subheader("📊 Current Dataset")

if st.session_state.dataset is not None:
    dataset = st.session_state.dataset

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Name", st.session_state.dataset_name)

    with col2:
        st.metric("Number of Cases", len(dataset._data))

    with col3:
        num_vars = len(dataset._meta.get('columns', {})) + len(dataset._meta.get('masks', {}))
        st.metric("Number of Variables", num_vars)

    # Show data preview
    st.markdown("#### Data Preview (First 10 Rows)")
    st.dataframe(dataset._data.head(10), use_container_width=True)

    # Show variable list
    with st.expander("📋 View All Variables"):
        try:
            columns = dataset.columns()
            masks = dataset.masks()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Regular Variables ({} total)**".format(len(columns)))
                st.write(columns)

            with col2:
                st.markdown("**Array Variables ({} total)**".format(len(masks)))
                st.write(masks)
        except Exception as e:
            st.error("Error displaying variables: {}".format(str(e)))

    # Navigation
    st.markdown("---")
    st.success("✅ Dataset ready! Navigate to **Data Explorer** or **Analysis** to continue.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Explore Data", use_container_width=True):
            st.switch_page("pages/02_Data_Explorer.py")
    with col2:
        if st.button("📊 Start Analysis", use_container_width=True):
            st.switch_page("pages/03_Analysis.py")

else:
    st.info("👆 No dataset loaded. Please select a data source above and load your data.")
