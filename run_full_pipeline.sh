#!/bin/bash
# Master script to run entire pipeline

echo "=================================="
echo "🥔 POTATO DISEASE SYSTEM - SETUP"
echo "=================================="

# Step 1: Install dependencies
echo ""
echo "📦 Step 1: Installing dependencies..."
pip install -r requirements_vit.txt

# Step 2: Download data
echo ""
echo "📥 Step 2: Downloading datasets..."
python scripts/1_download_data.py

if [ $? -ne 0 ]; then
    echo "❌ Data download failed. Please check Kaggle API setup."
    exit 1
fi

# Step 3: Preprocess weather data
echo ""
echo "🌤️  Step 3: Preprocessing weather data..."
python scripts/2_preprocess_weather.py

# Step 4: Train weather model
echo ""
echo "🌲 Step 4: Training weather model..."
python scripts/3_train_weather_model.py

# Step 5: Preprocess images
echo ""
echo "🖼️  Step 5: Preprocessing images..."
python scripts/4_preprocess_images.py

# Step 6: Train ViT (this takes longest)
echo ""
echo "🤖 Step 6: Training Vision Transformer (this may take 30-60 min)..."
python scripts/5_train_vit_model.py

# Step 7: Test models
echo ""
echo "🧪 Step 7: Testing models..."
python scripts/6_test_models.py

# Step 8: Launch dashboard
echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "🚀 Launch dashboard with:"
echo "   streamlit run app_potato_disease.py"
echo ""
