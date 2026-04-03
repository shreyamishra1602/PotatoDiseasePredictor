#!/bin/bash
# Package the project for Shreya's Windows machine
# Includes pre-trained models so she doesn't need to train

set -e

OUT="potato_disease_predictor"
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/data/models" "$OUT/data/processed" "$OUT/scripts" "$OUT/sample_test_images"

# Core app files
cp app_combined.py "$OUT/"
cp predict_image.py "$OUT/"
cp requirements_windows.txt "$OUT/"
cp requirements_vit.txt "$OUT/"
cp setup_windows.bat "$OUT/"
cp start.bat "$OUT/"
cp run_pipeline.sh "$OUT/"
cp shreya_guide.pdf "$OUT/"

# Pre-trained models (she doesn't need to retrain)
cp data/models/*.pkl "$OUT/data/models/" 2>/dev/null || true
cp data/models/*.pth "$OUT/data/models/" 2>/dev/null || true
cp data/models/*.json "$OUT/data/models/" 2>/dev/null || true

# Preprocessing artifacts
cp data/processed/*.pkl "$OUT/data/processed/" 2>/dev/null || true
cp data/processed/*.npy "$OUT/data/processed/" 2>/dev/null || true
cp data/processed/metadata.json "$OUT/data/processed/" 2>/dev/null || true

# Scripts (in case she wants to retrain)
cp scripts/*.py "$OUT/scripts/"

# Sample test images
cp -r sample_test_images/* "$OUT/sample_test_images/"

# Streamlit config
mkdir -p "$OUT/.streamlit"
cat > "$OUT/.streamlit/config.toml" << 'EOF'
[server]
headless = false

[browser]
gatherUsageStats = false
EOF

# README for Shreya
cat > "$OUT/README.txt" << 'EOF'
POTATO DISEASE PREDICTOR
========================

FIRST TIME SETUP:
1. Install Python from https://www.python.org/downloads/
   (Check "Add Python to PATH" during install!)
2. Double-click "setup_windows.bat"
3. Wait for it to finish

TO RUN:
- Double-click "start.bat"
- Browser will open automatically
- Use "Live Weather" to check blight risk
- Use "Leaf Scan" to upload potato leaf photos

SAMPLE IMAGES:
- sample_test_images/ folder has test photos you can try

Questions? Read shreya_guide.pdf
EOF

# Create zip
zip -r "$OUT.zip" "$OUT/" -x "*__pycache__*" "*.pyc"
echo ""
echo "Done! Created: $OUT.zip"
echo "Size: $(du -sh "$OUT.zip" | cut -f1)"
echo ""
echo "Send this zip to Shreya!"
