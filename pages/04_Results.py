#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantipy GUI - Results Page
View and export analysis results
"""

import streamlit as st
import pandas as pd
import quantipy as qp
import tempfile
import os

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

st.title("📈 Analysis Results")
st.markdown("View and export your analysis results.")

# Check if dataset and stack are loaded
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Please load a dataset first.")
    if st.button("📁 Go to Data Loader"):
        st.switch_page("pages/01_Data_Loader.py")
    st.stop()

if 'stack' not in st.session_state or st.session_state.stack is None:
    st.warning("⚠️ No analysis results available. Please run an analysis first.")
    if st.button("📊 Go to Analysis"):
        st.switch_page("pages/03_Analysis.py")
    st.stop()

dataset = st.session_state.dataset
stack = st.session_state.stack

st.markdown("---")

# Results display tabs
tab1, tab2, tab3 = st.tabs(["📊 View Results", "📥 Export Data", "📋 Summary"])

# Tab 1: View Results
with tab1:
    st.subheader("Browse Analysis Results")

    # Get stack description
    try:
        desc = stack.describe()
        st.markdown("#### Available Results")
        st.dataframe(desc, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Select Result to View")

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            # Get unique data keys
            data_keys = desc['data'].unique().tolist() if 'data' in desc.columns else []
            selected_data = st.selectbox("Data:", data_keys if data_keys else ['No data'])

        with col2:
            # Get filters for selected data
            if selected_data and selected_data != 'No data':
                filters = desc[desc['data'] == selected_data]['filter'].unique().tolist()
                selected_filter = st.selectbox("Filter:", filters if filters else ['no_filter'])
            else:
                selected_filter = 'no_filter'

        with col3:
            # Get x variables
            if selected_data and selected_data != 'No data':
                x_vars = desc[desc['data'] == selected_data]['x'].unique().tolist()
                selected_x = st.selectbox("X Variable:", x_vars if x_vars else ['None'])
            else:
                selected_x = 'None'

        # Y variable selection
        if selected_data and selected_data != 'No data' and selected_x and selected_x != 'None':
            y_vars = desc[(desc['data'] == selected_data) & (desc['x'] == selected_x)]['y'].unique().tolist()
            selected_y = st.selectbox("Y Variable:", y_vars if y_vars else ['@'])
        else:
            selected_y = '@'

        # View selection
        if (selected_data and selected_data != 'No data' and
            selected_x and selected_x != 'None'):
            try:
                link = stack[selected_data][selected_filter][selected_x][selected_y]
                view_keys = link.keys()
                selected_view = st.selectbox("View:", list(view_keys) if view_keys else ['No views'])
            except:
                selected_view = 'No views'
                st.warning("Unable to access link")

        st.markdown("---")

        # Display selected result
        if st.button("📊 Display Result", type="primary"):
            try:
                with st.spinner("Loading result..."):
                    link = stack[selected_data][selected_filter][selected_x][selected_y]

                    if selected_view and selected_view != 'No views':
                        result_df = link[selected_view]

                        st.markdown("#### Result: {} × {} [{}]".format(selected_x, selected_y, selected_view))

                        # Check if result has dataframe attribute
                        if hasattr(result_df, 'dataframe'):
                            display_df = result_df.dataframe
                        else:
                            display_df = result_df

                        st.dataframe(display_df, use_container_width=True)

                        # Download button
                        csv = display_df.to_csv()
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="result_{}_{}_{}.csv".format(selected_x, selected_y, selected_view.replace('|', '_')),
                            mime="text/csv"
                        )
                    else:
                        st.warning("No view selected or available")

            except Exception as e:
                st.error("Error displaying result: {}".format(str(e)))
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())

    except Exception as e:
        st.error("Error accessing results: {}".format(str(e)))
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())

# Tab 2: Export Data
with tab2:
    st.subheader("Export Results")

    st.markdown("Export your analysis results to various formats.")

    export_format = st.radio(
        "Select export format:",
        ["Excel (XLSX)", "Quantipy Format", "CSV (All Results)"],
        horizontal=True
    )

    st.markdown("---")

    # Excel Export
    if export_format == "Excel (XLSX)":
        st.markdown("#### Export to Excel")

        st.info("💡 Excel export uses the ExcelPainter to create formatted workbooks")

        excel_name = st.text_input("Workbook name:", value="quantipy_results")

        col1, col2 = st.columns(2)

        with col1:
            include_formats = st.checkbox("Include formatting", value=True)

        with col2:
            separate_sheets = st.checkbox("Separate sheets per variable", value=True)

        if st.button("📊 Generate Excel File", type="primary"):
            try:
                with st.spinner("Generating Excel file... This may take a moment."):
                    # Create temporary directory
                    temp_dir = tempfile.mkdtemp()
                    output_path = os.path.join(temp_dir, "{}.xlsx".format(excel_name))

                    # Use ExcelPainter
                    try:
                        painter = qp.ExcelPainter(stack)
                        painter.write_xlsx(output_path)

                        # Read the file and offer download
                        with open(output_path, 'rb') as f:
                            excel_data = f.read()

                        st.success("✅ Excel file generated!")

                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_data,
                            file_name="{}.xlsx".format(excel_name),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        # Cleanup
                        os.unlink(output_path)
                        os.rmdir(temp_dir)

                    except Exception as e:
                        st.error("Excel export error: {}".format(str(e)))
                        st.info("Note: ExcelPainter may require specific stack structure. Try using basic CSV export instead.")

            except Exception as e:
                st.error("Error generating Excel: {}".format(str(e)))
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())

    # Quantipy Format Export
    elif export_format == "Quantipy Format":
        st.markdown("#### Export to Quantipy Format")

        st.info("💡 Saves dataset as JSON metadata + CSV data files")

        export_name = st.text_input("Dataset name:", value=st.session_state.dataset_name)

        if st.button("💾 Export to Quantipy Format", type="primary"):
            try:
                with st.spinner("Exporting to Quantipy format..."):
                    # Create temporary files
                    temp_dir = tempfile.mkdtemp()
                    json_path = os.path.join(temp_dir, "{}.json".format(export_name))
                    csv_path = os.path.join(temp_dir, "{}.csv".format(export_name))

                    # Write files
                    dataset.write_quantipy(json_path, csv_path)

                    # Read files for download
                    with open(json_path, 'r') as f:
                        json_data = f.read()

                    with open(csv_path, 'r') as f:
                        csv_data = f.read()

                    st.success("✅ Files ready for download!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            label="📥 Download JSON Metadata",
                            data=json_data,
                            file_name="{}.json".format(export_name),
                            mime="application/json"
                        )

                    with col2:
                        st.download_button(
                            label="📥 Download CSV Data",
                            data=csv_data,
                            file_name="{}.csv".format(export_name),
                            mime="text/csv"
                        )

                    # Cleanup
                    os.unlink(json_path)
                    os.unlink(csv_path)
                    os.rmdir(temp_dir)

            except Exception as e:
                st.error("Error exporting: {}".format(str(e)))
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())

    # CSV Export All
    elif export_format == "CSV (All Results)":
        st.markdown("#### Export All Results to CSV")

        st.info("💡 Exports each result as a separate CSV file in a zip archive")

        if st.button("📦 Generate CSV Archive", type="primary"):
            st.warning("This feature requires additional implementation for ZIP creation")
            st.info("For now, use the 'View Results' tab to download individual results as CSV")

# Tab 3: Summary
with tab3:
    st.subheader("Analysis Summary")

    try:
        desc = stack.describe()

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Links", len(desc))

        with col2:
            num_x_vars = desc['x'].nunique() if 'x' in desc.columns else 0
            st.metric("X Variables", num_x_vars)

        with col3:
            num_y_vars = desc['y'].nunique() if 'y' in desc.columns else 0
            st.metric("Y Variables", num_y_vars)

        with col4:
            num_views = desc['view'].nunique() if 'view' in desc.columns else 0
            st.metric("View Types", num_views)

        st.markdown("---")

        # Breakdown by variable
        st.markdown("#### Results by X Variable")
        if 'x' in desc.columns:
            x_summary = desc.groupby('x').size().reset_index(name='count')
            st.dataframe(x_summary, use_container_width=True)

        st.markdown("#### Results by Y Variable")
        if 'y' in desc.columns:
            y_summary = desc.groupby('y').size().reset_index(name='count')
            st.dataframe(y_summary, use_container_width=True)

        st.markdown("#### Results by View Type")
        if 'view' in desc.columns:
            view_summary = desc.groupby('view').size().reset_index(name='count')
            st.dataframe(view_summary, use_container_width=True)

        st.markdown("---")

        # Full description table
        st.markdown("#### Complete Results Listing")
        st.dataframe(desc, use_container_width=True, height=400)

        # Download summary
        csv = desc.to_csv()
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv,
            file_name="analysis_summary.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Error generating summary: {}".format(str(e)))

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
    if st.button("📊 Back to Analysis", use_container_width=True):
        st.switch_page("pages/03_Analysis.py")
