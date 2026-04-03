#!/bin/bash
# Potato Disease Prediction - Full Pipeline Runner
# Run from the project root: ./run_pipeline.sh

set -e

VENV_DIR=".venv"

echo "=============================================="
echo "  POTATO DISEASE PREDICTION PIPELINE"
echo "=============================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[0/7] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Created .venv"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python)"

# Install dependencies
echo ""
echo "[1/7] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements_vit.txt -q

# Download data
echo ""
echo "[2/7] Downloading datasets from Kaggle..."
python scripts/1_download_data.py

# Preprocess weather
echo ""
echo "[3/7] Preprocessing weather data..."
python scripts/2_preprocess_weather.py

# Train weather model
echo ""
echo "[4/7] Training weather model..."
python scripts/3_train_weather_model.py

# Preprocess images
echo ""
echo "[5/7] Preprocessing images..."
python scripts/4_preprocess_images.py

# Train ViT
echo ""
echo "[6/7] Training ViT model (this takes a while)..."
python scripts/5_train_vit_model.py

# Test
echo ""
echo "[7/7] Testing models..."
python scripts/6_test_models.py

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE!"
echo "=============================================="
echo ""
echo "To launch the dashboard:"
echo "  source .venv/bin/activate"
echo "  streamlit run app_combined.py"
echo ""
