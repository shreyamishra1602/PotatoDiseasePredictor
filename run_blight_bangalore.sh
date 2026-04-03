#!/bin/bash
# Potato Blight Risk Dashboard - Bangalore Demo

echo "========================================="
echo "  🥔 Potato Blight Risk Dashboard"
echo "  📍 Bangalore Edition"
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
echo "🚀 Starting Bangalore Potato Blight Dashboard..."
echo "   Open http://localhost:8501 in your browser"
echo "   Shows real-time weather and blight risk for Bangalore"
echo "   Press Ctrl+C to stop"
echo ""

$PY -m streamlit run potato_blight_demo.py