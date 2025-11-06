#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantipy GUI - Data Explorer Page
Explore dataset variables, metadata, and crosstabs
"""

import streamlit as st
import pandas as pd
import quantipy as qp
from quantipy.core.tools.dp.prep import frequency, crosstab
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

st.title("🔍 Data Explorer")
st.markdown("Explore your dataset variables, metadata, and create crosstabulations.")

# Check if dataset is loaded
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please load a dataset first.")
    if st.button("📁 Go to Data Loader"):
        st.switch_page("pages/01_Data_Loader.py")
    st.stop()

dataset = st.session_state.dataset

# Sidebar for variable selection
st.sidebar.header("🔧 Explorer Options")

explorer_mode = st.sidebar.radio(
    "Select Exploration Mode:",
    ["Variable Browser", "Frequency Tables", "Crosstabs", "Metadata Viewer"]
)

st.markdown("---")

# Variable Browser Mode
if explorer_mode == "Variable Browser":
    st.subheader("📋 Variable Browser")

    # Get all variables
    try:
        columns = dataset.columns()
        masks = dataset.masks()
        all_vars = columns + masks
    except:
        all_vars = dataset._data.columns.tolist()
        columns = all_vars
        masks = []

    # Variable type filter
    var_type_filter = st.radio(
        "Filter by Type:",
        ["All Variables", "Regular Variables Only", "Array Variables Only"],
        horizontal=True
    )

    if var_type_filter == "Regular Variables Only":
        display_vars = columns
    elif var_type_filter == "Array Variables Only":
        display_vars = masks
    else:
        display_vars = all_vars

    st.info("Showing {} variables".format(len(display_vars)))

    # Search functionality
    search = st.text_input("🔍 Search variables", "")
    if search:
        display_vars = [v for v in display_vars if search.lower() in v.lower()]
        st.caption("Found {} matching variables".format(len(display_vars)))

    # Display variables in a table
    if display_vars:
        var_info = []
        for var in display_vars:
            try:
                # Get variable type
                if var in dataset._meta.get('columns', {}):
                    var_type = dataset._meta['columns'][var].get('type', 'unknown')
                    var_text = dataset._meta['columns'][var].get('text', {}).get(dataset.text_key, var)
                elif var in dataset._meta.get('masks', {}):
                    var_type = 'array'
                    var_text = dataset._meta['masks'][var].get('text', {}).get(dataset.text_key, var)
                else:
                    var_type = 'unknown'
                    var_text = var

                # Get unique count
                if var in dataset._data.columns:
                    unique_count = dataset._data[var].nunique()
                else:
                    unique_count = '-'

                var_info.append({
                    'Variable': var,
                    'Label': var_text,
                    'Type': var_type,
                    'Unique Values': unique_count
                })
            except:
                var_info.append({
                    'Variable': var,
                    'Label': var,
                    'Type': 'unknown',
                    'Unique Values': '-'
                })

        df_vars = pd.DataFrame(var_info)
        st.dataframe(df_vars, use_container_width=True, height=400)

        # Select a variable to view details
        st.markdown("---")
        st.subheader("📊 Variable Details")

        selected_var = st.selectbox("Select a variable to view details:", display_vars)

        if selected_var:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Variable Information**")
                try:
                    meta_info = dataset.meta(selected_var)
                    st.json(meta_info)
                except Exception as e:
                    st.info("Metadata not available: {}".format(str(e)))

            with col2:
                st.markdown("**Value Distribution**")
                try:
                    if selected_var in dataset._data.columns:
                        freq_df = dataset._data[selected_var].value_counts().reset_index()
                        freq_df.columns = ['Value', 'Count']
                        st.dataframe(freq_df, use_container_width=True)

                        # Simple bar chart
                        if len(freq_df) <= 20:  # Only show chart for reasonable number of values
                            fig = px.bar(freq_df.head(15), x='Value', y='Count',
                                       title='Top 15 Values')
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Error showing distribution: {}".format(str(e)))

    else:
        st.info("No variables to display")

# Frequency Tables Mode
elif explorer_mode == "Frequency Tables":
    st.subheader("📊 Frequency Tables")

    try:
        columns = dataset.columns()
        all_vars = columns + dataset.masks()
    except:
        all_vars = dataset._data.columns.tolist()

    selected_var = st.selectbox("Select variable for frequency table:", all_vars)

    col1, col2 = st.columns([3, 1])

    with col2:
        show_text = st.checkbox("Show Text Labels", value=True)
        show_counts = st.checkbox("Show Counts", value=True)
        show_pct = st.checkbox("Show Percentages", value=True)

    if st.button("Generate Frequency Table", type="primary"):
        try:
            with st.spinner("Generating frequency table..."):
                # Use dataset.crosstab for single variable frequency
                result = dataset.crosstab(selected_var, text=show_text)
                st.dataframe(result, use_container_width=True)

                # Create visualization
                if selected_var in dataset._data.columns:
                    freq_data = dataset._data[selected_var].value_counts().head(15)

                    fig = go.Figure(data=[
                        go.Bar(x=freq_data.index.astype(str), y=freq_data.values)
                    ])
                    fig.update_layout(
                        title='Frequency Distribution: {}'.format(selected_var),
                        xaxis_title='Value',
                        yaxis_title='Count',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Error generating frequency table: {}".format(str(e)))
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

# Crosstabs Mode
elif explorer_mode == "Crosstabs":
    st.subheader("📈 Crosstabulations")

    st.markdown("Create cross-tabulations between two variables")

    try:
        columns = dataset.columns()
        all_vars = columns + dataset.masks()
    except:
        all_vars = dataset._data.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        x_var = st.selectbox("Select X Variable (columns):", all_vars, key='x_var')

    with col2:
        y_var = st.selectbox("Select Y Variable (rows):", ['@'] + all_vars, key='y_var')
        st.caption("@ = base (total)")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_text = st.checkbox("Show Text Labels", value=True, key='ct_text')

    with col2:
        show_counts = st.checkbox("Show Counts", value=True, key='ct_counts')

    with col3:
        pct_type = st.selectbox("Percentage Type:", ["None", "Column %", "Row %", "Total %"])

    if st.button("Generate Crosstab", type="primary"):
        try:
            with st.spinner("Generating crosstab..."):
                result = dataset.crosstab(x_var, y_var, text=show_text)
                st.dataframe(result, use_container_width=True)

                # Download option
                csv = result.to_csv()
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="crosstab_{}_{}.csv".format(x_var, y_var),
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Error generating crosstab: {}".format(str(e)))
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())

# Metadata Viewer Mode
elif explorer_mode == "Metadata Viewer":
    st.subheader("🗂️ Metadata Viewer")

    st.markdown("View the complete metadata structure of your dataset")

    metadata_section = st.selectbox(
        "Select metadata section:",
        ["Overview", "Columns", "Masks", "Sets", "Library"]
    )

    if metadata_section == "Overview":
        st.markdown("### Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dataset Name", st.session_state.dataset_name)
            st.metric("Number of Cases", len(dataset._data))

        with col2:
            num_cols = len(dataset._meta.get('columns', {}))
            st.metric("Regular Variables", num_cols)
            num_masks = len(dataset._meta.get('masks', {}))
            st.metric("Array Variables", num_masks)

        with col3:
            text_key = dataset.text_key if dataset.text_key else 'Not set'
            st.metric("Text Key", text_key)
            num_sets = len(dataset._meta.get('sets', {}))
            st.metric("Variable Sets", num_sets)

        st.markdown("### Data Shape")
        st.write("Rows: {}, Columns: {}".format(dataset._data.shape[0], dataset._data.shape[1]))

    elif metadata_section == "Columns":
        st.markdown("### Column Metadata")
        columns_meta = dataset._meta.get('columns', {})

        if columns_meta:
            var_name = st.selectbox("Select variable:", list(columns_meta.keys()))
            if var_name:
                st.json(columns_meta[var_name])
        else:
            st.info("No column metadata available")

    elif metadata_section == "Masks":
        st.markdown("### Array/Mask Metadata")
        masks_meta = dataset._meta.get('masks', {})

        if masks_meta:
            mask_name = st.selectbox("Select array:", list(masks_meta.keys()))
            if mask_name:
                st.json(masks_meta[mask_name])
        else:
            st.info("No mask metadata available")

    elif metadata_section == "Sets":
        st.markdown("### Variable Sets")
        sets_meta = dataset._meta.get('sets', {})

        if sets_meta:
            set_name = st.selectbox("Select set:", list(sets_meta.keys()))
            if set_name:
                st.json(sets_meta[set_name])
        else:
            st.info("No sets metadata available")

    elif metadata_section == "Library":
        st.markdown("### Value Library")
        lib_meta = dataset._meta.get('lib', {})

        if lib_meta:
            st.json(lib_meta)
        else:
            st.info("No library metadata available")

    # Full metadata export
    st.markdown("---")
    if st.button("📥 Download Full Metadata as JSON"):
        import json
        meta_json = json.dumps(dataset._meta, indent=2)
        st.download_button(
            label="Download JSON",
            data=meta_json,
            file_name="{}_metadata.json".format(st.session_state.dataset_name),
            mime="application/json"
        )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📁 Back to Data Loader", use_container_width=True):
        st.switch_page("pages/01_Data_Loader.py")

with col2:
    if st.button("📊 Go to Analysis", use_container_width=True):
        st.switch_page("pages/03_Analysis.py")

with col3:
    if st.button("📈 View Results", use_container_width=True):
        st.switch_page("pages/04_Results.py")
