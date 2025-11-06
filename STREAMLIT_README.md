# Quantipy Streamlit GUI

A user-friendly web interface for Quantipy data analysis, built with Streamlit.

## Overview

This Streamlit application provides an interactive graphical user interface for Quantipy's data processing and analysis capabilities. It's designed for researchers and analysts working with survey and market research data.

## Features

### 📁 Data Loader
- Load example datasets for quick exploration
- Import Quantipy files (JSON metadata + CSV data)
- Upload CSV files with automatic metadata inference
- Import SPSS .sav files
- View dataset summary and preview

### 🔍 Data Explorer
- **Variable Browser**: Browse and search all dataset variables
- **Frequency Tables**: Generate frequency distributions for individual variables
- **Crosstabs**: Create cross-tabulations between variables
- **Metadata Viewer**: Inspect complete dataset metadata structure
- Interactive charts and visualizations

### 📊 Analysis
- Create and manage multiple batch definitions
- Configure X variables (columns) and Y variables (rows)
- Define analysis specifications
- Run aggregations (counts, percentages, means, etc.)
- Generate comprehensive analysis stacks

### 📈 Results
- Browse all analysis results interactively
- View individual crosstabs and statistics
- Export to Excel (XLSX) with formatting
- Export to Quantipy format (JSON + CSV)
- Download individual results as CSV
- View analysis summary and statistics

## Installation

### Prerequisites

1. **Python 2.7** (as required by Quantipy)
2. **Conda** (recommended for managing Python 2.7 environment)

### Setup

1. Create and activate a Python 2.7 environment:

```bash
# Create environment
conda create -n quantipy_gui python=2.7 numpy==1.11.3 scipy==0.18.1

# Activate environment
conda activate quantipy_gui
```

2. Install Quantipy and dependencies:

```bash
pip install -r requirements_dev.txt
```

3. Install Streamlit and additional dependencies:

```bash
pip install -r requirements_streamlit.txt
```

## Running the Application

### Start the Streamlit App

From the project root directory, run:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Example Data

1. Click "Load Example Data & Explore" on the home page, or
2. Navigate to the Data Loader page and click "Load Example Dataset"

This will load the built-in example dataset (Example Data A) from the tests folder.

## Application Structure

```
quantipy_GUI/
├── streamlit_app.py              # Main application entry point
├── pages/
│   ├── 01_Data_Loader.py         # Data import page
│   ├── 02_Data_Explorer.py       # Data exploration page
│   ├── 03_Analysis.py            # Analysis configuration page
│   └── 04_Results.py             # Results viewing and export page
├── requirements_streamlit.txt     # Streamlit-specific requirements
└── STREAMLIT_README.md           # This file
```

## Usage Guide

### Basic Workflow

1. **Load Data**
   - Go to "Data Loader" page
   - Choose a data source (Example, Quantipy files, CSV, or SPSS)
   - Upload your files or load example data

2. **Explore Data**
   - Navigate to "Data Explorer"
   - Browse variables and view metadata
   - Create frequency tables and crosstabs
   - Examine data distributions

3. **Configure Analysis**
   - Go to "Analysis" page
   - Create a new batch
   - Add X variables (columns) and Y variables (rows)
   - Select aggregation types

4. **Run Analysis**
   - Click "Run Analysis" to generate results
   - Wait for processing to complete

5. **View Results**
   - Navigate to "Results" page
   - Browse individual results
   - Export to Excel, CSV, or Quantipy format

### Tips

- **Session Persistence**: Your loaded dataset and analysis results persist across pages during your session
- **Navigation**: Use the page links in the sidebar or navigation buttons at the bottom of each page
- **Example Data**: Start with the example dataset to familiarize yourself with the interface
- **Error Details**: Most error messages include expandable details for troubleshooting

## Limitations

### Python 2.7 Compatibility

This application is built for Python 2.7 to maintain compatibility with Quantipy. Some modern Streamlit features may not be available.

### Performance

- Large datasets (>100,000 rows) may take longer to process
- Complex batch configurations may require significant processing time
- Excel export with ExcelPainter can be memory-intensive

### Known Issues

- Some advanced Quantipy features (statistical tests, complex filters) are simplified in the GUI
- Chain and Cluster functionality not yet exposed in the interface
- ZIP archive export for multiple CSV files requires additional implementation

## Troubleshooting

### App Won't Start

```bash
# Verify Streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements_streamlit.txt
pip install -r requirements_dev.txt
```

### Display Issues

- Clear Streamlit cache: Press 'C' in the app or use the menu
- Restart the Streamlit server
- Check browser compatibility (Chrome, Firefox, Safari recommended)

### Memory Issues with Large Datasets

- Use data subsets for exploration
- Close other applications
- Consider upgrading your system RAM

## Advanced Features

### Custom Filters

While basic filtering is available through the interface, complex filters can be defined in the Analysis page using Quantipy's logic expressions.

### Weighting

Weight variables can be selected in the Analysis configuration. The weights will be applied during aggregation.

### Export Formats

- **Excel**: Uses Quantipy's ExcelPainter for formatted workbooks
- **Quantipy Format**: Preserves all metadata for future use
- **CSV**: Universal format for use in other tools

## Contributing

This GUI is part of the Quantipy project. For issues or suggestions:

1. Report bugs via GitHub issues
2. Submit feature requests
3. Contribute improvements via pull requests

## Support

- **Quantipy Documentation**: http://quantipy.readthedocs.io/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Python 3 Version**: https://github.com/quantipy/quantipy3

## License

Same license as the main Quantipy project.

---

**Note**: This GUI is designed for Python 2.7 compatibility. For Python 3 support, consider using the Quantipy3 fork.
