# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantipy is a Python 2.7-based data processing, analysis and reporting library for survey and market research data (people data). It extends pandas and numpy with specialized features for multiple choice variables, weighted analysis, metadata-driven operations, and exports to various formats.

**Note**: This is the Python 2.7 version. A Python 3 port exists in a separate repository.

## Development Setup

### Creating Development Environment

**Windows:**
```bash
conda create -n envqp python=2.7 numpy==1.11.3 scipy==0.18.1
conda activate envqp
pip install -r requirements_dev.txt
```

**Linux:**
```bash
conda create -n envqp python=2.7
conda activate envqp
pip install -r requirements_dev.txt
```

Or use the provided script:
```bash
bash install_dev.sh
```

### Key Dependencies
- Python 2.7.8
- numpy==1.11.3
- scipy==0.18.1
- pandas==0.19.2
- Additional: xlsxwriter, python-pptx, lxml, ftfy, xmltodict

## Streamlit GUI

A web-based graphical interface is available for Quantipy, providing interactive access to core functionality without requiring code.

### Running the Streamlit App

```bash
# Install Streamlit dependencies
pip install -r requirements_streamlit.txt

# Start the application
streamlit run streamlit_app.py

# Or use the launcher script
./run_streamlit.sh
```

The app will open at `http://localhost:8501`

### GUI Architecture

The Streamlit app uses a multi-page architecture with session state for data persistence:

- **`streamlit_app.py`**: Main entry point and home page
- **`pages/01_Data_Loader.py`**: Data import from multiple formats (Quantipy, CSV, SPSS)
- **`pages/02_Data_Explorer.py`**: Variable browsing, frequencies, crosstabs, metadata viewing
- **`pages/03_Analysis.py`**: Batch creation and analysis configuration
- **`pages/04_Results.py`**: Results viewing and export (Excel, CSV, Quantipy format)

Key implementation notes:
- Session state (`st.session_state`) maintains dataset, stack, and batches across pages
- Uses Plotly for interactive visualizations
- Temporary files for upload/download operations (cleaned up after use)
- Compatible with Python 2.7 Streamlit versions

See `STREAMLIT_README.md` for detailed GUI documentation.

## Testing

### Run All Tests
```bash
python -m unittest discover
```

Or with pytest:
```bash
pytest
```

### Run Tests with Coverage
```bash
coverage run -m unittest discover
coverage html
# View reports in htmlcov/index.html
```

### Run Tests with Multiple Cores
```bash
pytest -n auto
```

### Auto-Run Tests on File Changes
```bash
python autotests.py
```

## Core Architecture

Quantipy uses a hierarchical object structure for managing survey data analysis:

### Primary Objects Hierarchy

**DataSet** → **Batch** → **Stack** → **Link** → **View**

1. **DataSet** (`quantipy/core/dataset.py`)
   - Main container for case data (pandas DataFrame) and metadata (JSON structure)
   - Handles data import/export, variable creation, recoding, and transformations
   - Metadata format describes variables, their types (single, delimited, array), and values
   - Methods: `derive()`, `recode()`, `merge()`, `crosstab()`, `variables()`, `meta()`

2. **Batch** (`quantipy/core/batch.py`)
   - Subclass of DataSet for defining analysis specifications
   - Structures which variables to cross-tabulate (x vs y variables)
   - Stores batch definitions in dataset metadata under `_meta['sets']['batches']`
   - Methods: `add_x()`, `add_y()`, `add_filter()`

3. **Stack** (`quantipy/core/stack.py`)
   - Nested dictionary container holding Link objects with View aggregations
   - Structure: `stack[data_key][filter][x_variable][y_variable][view_key]`
   - Created by calling `dataset.populate()` based on Batch definitions
   - Methods: `add_data()`, `add_link()`, `aggregate()`, `add_stats()`, `describe()`

4. **Link** (`quantipy/core/link.py`)
   - Subclassed dictionary representing a single data/filter/x/y relationship
   - Each Link contains multiple View aggregations of the same variable pairing
   - Accessed as: `link = stack[data_key][filter][x][y]`

5. **View** (`quantipy/core/view.py`)
   - Represents a specific aggregation/analysis (counts, percentages, means, tests)
   - Stored as pandas DataFrames within Link objects
   - View types: frequency counts, column/row percentages, means, statistical tests

6. **Chain** (`quantipy/core/chain.py`)
   - Container for ordered Link definitions and associated Views
   - Used for organizing and concatenating multiple analyses along an axis
   - Supports serialization to/from `.chain` files using cPickle

7. **Cluster** (`quantipy/core/cluster.py`)
   - Higher-level container for managing multiple Chain objects
   - Used for structured reporting and analysis workflows

### Key Supporting Modules

**Data Processing Tools** (`quantipy/core/tools/dp/`)
- `io.py`: Import/export functions for all supported formats
- `prep.py`: Data preparation utilities (merge, recode, frequency, crosstab)
- `query.py`: Logic-based filtering and subsetting
- `spss/`: SPSS .sav file reader/writer (uses savReaderWriter)
- `dimensions/`: Dimensions .ddf/.mdd file support
- `decipher/`: Decipher format support
- `ascribe/`: Ascribe format support

**View Tools** (`quantipy/core/tools/view/`)
- `agg.py`: Aggregation methods
- `logic.py`: Logical operators (has_any, has_all, is_gt, union, intersection)
- `query.py`: View-level filtering

**Export Builders** (`quantipy/core/builds/`)
- `excel/excel_painter.py`: ExcelPainter for XLSX exports with formatting
- `powerpoint/pptx_painter.py`: PowerPointPainter for PPTX chart/table exports

**Weighting** (`quantipy/core/weights/`)
- `rim.py`: Rim weighting (iterative proportional fitting)
- `weight_engine.py`: Weight computation engine

**Analysis Engine** (`quantipy/core/quantify/`)
- `engine.py`: Quantity and Test classes for advanced aggregations and statistical tests

### Variable Types

Quantipy distinguishes between three core variable types in metadata:

- **single**: Single-choice categorical variables
- **delimited**: Multiple-choice variables (stored as delimited strings like "1;3;5;")
- **array**: Grids/matrices with multiple items sharing the same response scale
  - Array items stored as separate columns but grouped in `_meta['masks']`

### Metadata Structure

Metadata is stored in `dataset._meta` as a nested dictionary:
- `_meta['columns']`: Column-level metadata (type, text, values)
- `_meta['masks']`: Array/grid definitions
- `_meta['sets']`: Named sets including batch definitions
- `_meta['lib']`: Shared value definitions

## Common Workflow Patterns

### Typical Analysis Workflow
1. Load data: `dataset = qp.DataSet('name'); dataset.read_quantipy(json_path, csv_path)`
2. Create batch: `batch = dataset.add_batch('batch_name')`
3. Define axes: `batch.add_x(['q1', 'q2']); batch.add_y(['gender', 'age'])`
4. Populate stack: `stack = dataset.populate()`
5. Add aggregations: `stack.aggregate(['counts', 'c%'])`
6. Export: `painter = qp.ExcelPainter(stack); painter.write_xlsx(path)`

### Variable Manipulation
- Use `dataset.derive()` to create new variables from existing ones
- Use `dataset.recode()` to remap variable values
- Use `frange()` helper for range specifications: `frange('1-5, 97, 99')`

### Accessing Results
```python
# Get specific link
link = stack[data_key][filter][x_var][y_var]

# Get specific view from link
df = link[view_key]

# Use Quantity engine for custom aggregations
q = qp.Quantity(link)
q.count()  # Returns grouped DataFrame
```

## File I/O Formats

Quantipy supports reading from:
- Native Quantipy (.json metadata + .csv data)
- SPSS .sav files
- Dimensions .ddf/.mdd files
- Decipher tab-delimited files
- Ascribe files

Quantipy supports exporting to:
- Native Quantipy format
- SPSS .sav
- Dimensions .ddf/.mdd
- Excel .xlsx (with ExcelPainter)
- PowerPoint .pptx (with PowerPointPainter)

Use functions from `quantipy.core.tools.dp.io` for all I/O operations.

## Code Style Notes

- This is Python 2.7 code - print statements, not print functions
- Uses `cPickle` for serialization
- Relies on older pandas 0.19.2 API (e.g., `.ix` accessor instead of `.loc`/`.iloc`)
- Extensive use of nested dictionaries and defaultdict for data structures
