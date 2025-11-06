#!/bin/bash

# Quantipy Streamlit GUI Launcher
# This script starts the Streamlit application

echo "========================================="
echo "Quantipy Streamlit GUI"
echo "========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit is not installed!"
    echo ""
    echo "Please install requirements:"
    echo "  pip install -r requirements_streamlit.txt"
    echo ""
    exit 1
fi

# Check if quantipy is available
python -c "import quantipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Quantipy is not installed!"
    echo ""
    echo "Please install requirements:"
    echo "  pip install -r requirements_dev.txt"
    echo ""
    exit 1
fi

echo "✅ Dependencies found"
echo ""
echo "Starting Streamlit application..."
echo "The app will open in your default browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run streamlit_app.py
