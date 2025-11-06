#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantipy GUI - Analysis Page
Create batches, define analyses, and run aggregations
"""

import streamlit as st
import pandas as pd
import quantipy as qp

st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")

st.title("📊 Analysis")
st.markdown("Create batch definitions and run analyses on your dataset.")

# Check if dataset is loaded
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please load a dataset first.")
    if st.button("📁 Go to Data Loader"):
        st.switch_page("pages/01_Data_Loader.py")
    st.stop()

dataset = st.session_state.dataset

# Initialize session state for analysis
if 'batches' not in st.session_state:
    st.session_state.batches = {}
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None
if 'stack' not in st.session_state:
    st.session_state.stack = None

# Get available variables
try:
    columns = dataset.columns()
    masks = dataset.masks()
    all_vars = columns + masks
except:
    all_vars = dataset._data.columns.tolist()

st.markdown("---")

# Analysis workflow tabs
tab1, tab2, tab3 = st.tabs(["📋 Create Batch", "⚙️ Configure Analysis", "▶️ Run Analysis"])

# Tab 1: Create Batch
with tab1:
    st.subheader("Create or Select Batch")

    # Batch management
    col1, col2 = st.columns([2, 1])

    with col1:
        batch_name = st.text_input("Batch Name", value="batch_1")

    with col2:
        if st.button("➕ Create New Batch", type="primary"):
            try:
                if batch_name in st.session_state.batches:
                    st.warning("Batch '{}' already exists. Updating...".format(batch_name))

                batch = dataset.add_batch(batch_name)
                st.session_state.batches[batch_name] = batch
                st.session_state.current_batch = batch_name
                st.success("✅ Batch '{}' created!".format(batch_name))
                st.rerun()
            except Exception as e:
                st.error("Error creating batch: {}".format(str(e)))

    # Show existing batches
    if st.session_state.batches:
        st.markdown("### Existing Batches")

        batch_list = list(st.session_state.batches.keys())
        selected_batch = st.selectbox("Select a batch to work with:", batch_list)

        if selected_batch:
            st.session_state.current_batch = selected_batch
            batch = st.session_state.batches[selected_batch]

            # Show batch configuration
            st.markdown("#### Batch Configuration")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**X Variables (Columns)**")
                if hasattr(batch, 'x_variables') and batch.x_variables:
                    st.write(batch.x_variables)
                else:
                    st.info("No x variables defined")

            with col2:
                st.markdown("**Y Variables (Rows)**")
                if hasattr(batch, 'y_variables') and batch.y_variables:
                    st.write(batch.y_variables)
                else:
                    st.info("No y variables defined")

    else:
        st.info("👆 No batches created yet. Create a new batch to start.")

# Tab 2: Configure Analysis
with tab2:
    st.subheader("Configure Batch Variables")

    if st.session_state.current_batch is None:
        st.warning("⚠️ Please create or select a batch first.")
    else:
        batch = st.session_state.batches[st.session_state.current_batch]
        st.info("Configuring batch: **{}**".format(st.session_state.current_batch))

        # X Variables (columns)
        st.markdown("### X Variables (Column Variables)")
        st.caption("These variables will appear as columns in your crosstabs")

        x_vars = st.multiselect(
            "Select X variables:",
            all_vars,
            key='x_vars_select',
            help="Choose variables to appear as columns"
        )

        if st.button("Add X Variables", key='add_x'):
            if x_vars:
                try:
                    batch.add_x(x_vars)
                    st.success("✅ Added {} x variables".format(len(x_vars)))
                    st.rerun()
                except Exception as e:
                    st.error("Error adding x variables: {}".format(str(e)))
            else:
                st.warning("Please select at least one variable")

        st.markdown("---")

        # Y Variables (rows)
        st.markdown("### Y Variables (Row Variables)")
        st.caption("These variables will appear as rows in your crosstabs")

        # Add @ option for base
        y_var_options = ['@'] + all_vars

        y_vars = st.multiselect(
            "Select Y variables:",
            y_var_options,
            key='y_vars_select',
            help="Choose variables to appear as rows. '@' represents the base/total."
        )

        if st.button("Add Y Variables", key='add_y'):
            if y_vars:
                try:
                    batch.add_y(y_vars)
                    st.success("✅ Added {} y variables".format(len(y_vars)))
                    st.rerun()
                except Exception as e:
                    st.error("Error adding y variables: {}".format(str(e)))
            else:
                st.warning("Please select at least one variable")

        st.markdown("---")

        # Filters (optional)
        st.markdown("### Filters (Optional)")
        st.caption("Apply filters to subset your data")

        with st.expander("Add Filter (Advanced)"):
            st.info("Filter functionality available - requires logic expressions")
            filter_alias = st.text_input("Filter Name/Alias")
            st.text_area(
                "Filter Logic",
                help="Example: {'gender': [1]} for males only",
                placeholder="Enter filter logic as dict"
            )
            st.button("Add Filter", disabled=True, help="Advanced feature - coming soon")

        # Weights (optional)
        st.markdown("### Weighting (Optional)")

        weight_options = ['None'] + all_vars
        weight_var = st.selectbox("Select weight variable:", weight_options)

        if weight_var and weight_var != 'None':
            st.info("Weight variable: **{}** will be applied".format(weight_var))

# Tab 3: Run Analysis
with tab3:
    st.subheader("Run Analysis & Generate Results")

    if st.session_state.current_batch is None:
        st.warning("⚠️ Please create and configure a batch first.")
    else:
        batch = st.session_state.batches[st.session_state.current_batch]
        st.info("Ready to analyze batch: **{}**".format(st.session_state.current_batch))

        # Show current configuration
        st.markdown("#### Current Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**X Variables**")
            if hasattr(batch, 'x_variables') and batch.x_variables:
                for var in batch.x_variables:
                    st.write("- " + var)
            else:
                st.caption("None defined")

        with col2:
            st.markdown("**Y Variables**")
            if hasattr(batch, 'y_variables') and batch.y_variables:
                for var in batch.y_variables:
                    st.write("- " + var)
            else:
                st.caption("None defined")

        st.markdown("---")

        # Analysis options
        st.markdown("#### Analysis Options")

        col1, col2 = st.columns(2)

        with col1:
            aggregations = st.multiselect(
                "Select aggregations:",
                ['counts', 'c%', 'r%', 'mean', 'median', 'stddev'],
                default=['counts', 'c%'],
                help="Choose which statistics to calculate"
            )

        with col2:
            add_stats = st.checkbox("Add statistical tests", value=False)
            if add_stats:
                st.caption("⚠️ Advanced feature")

        # Run button
        st.markdown("---")

        if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
            # Check if batch is configured
            if not hasattr(batch, 'x_variables') or not batch.x_variables:
                st.error("❌ Please add X variables first")
            elif not hasattr(batch, 'y_variables') or not batch.y_variables:
                st.error("❌ Please add Y variables first")
            else:
                try:
                    with st.spinner("Running analysis... This may take a moment."):
                        # Create stack from batch
                        stack = dataset.populate()

                        # Add aggregations
                        if aggregations:
                            stack.aggregate(aggregations, verbose=False)

                        # Store in session state
                        st.session_state.stack = stack

                        st.success("✅ Analysis complete!")
                        st.balloons()

                        # Show quick summary
                        st.markdown("#### Analysis Summary")
                        try:
                            desc = stack.describe()
                            st.dataframe(desc.head(20), use_container_width=True)

                            st.info("📊 {} link(s) created. Go to **Results** page to view details.".format(len(desc)))
                        except Exception as e:
                            st.warning("Summary not available: {}".format(str(e)))

                        # Navigation to results
                        if st.button("📈 View Results", type="primary"):
                            st.switch_page("pages/04_Results.py")

                except Exception as e:
                    st.error("Error running analysis: {}".format(str(e)))
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())

        # Show existing stack info
        if st.session_state.stack is not None:
            st.markdown("---")
            st.success("✅ Analysis results are available")

            if st.button("📈 Go to Results Page", use_container_width=True):
                st.switch_page("pages/04_Results.py")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📁 Back to Data Loader", use_container_width=True):
        st.switch_page("pages/01_Data_Loader.py")

with col2:
    if st.button("🔍 Back to Explorer", use_container_width=True):
        st.switch_page("pages/02_Data_Explorer.py")

with col3:
    if st.button("📈 View Results", use_container_width=True):
        st.switch_page("pages/04_Results.py")
