#!/bin/bash
# Potato Blight Risk Prediction — Mac/Linux launcher

echo "========================================="
echo "  🥔 Potato Blight Risk Prediction"
echo "========================================="
echo ""

# Check for Python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python not found!"
    echo "Install it from https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Found $($PY --version)"

# Install dependencies
echo "📦 Installing dependencies..."
$PY -m pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "⚠️  pip install failed, trying with --break-system-packages..."
    $PY -m pip install -r requirements.txt --quiet --break-system-packages
fi

echo ""
echo "🚀 Starting the potato blight prediction system..."
echo "   Open http://localhost:8501 in your browser"
echo "   Press Ctrl+C to stop"
echo ""

$PY -m streamlit run potato_blight_prediction.py