#!/usr/bin/env python3
"""
Model Testing Script - Test both Weather and ViT models
Torch/timm are lazy-imported only for ViT test to avoid MPS+pickle segfaults.
"""

import numpy as np
import pickle
from pathlib import Path
import json


def get_device():
    """Get best available device"""
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def test_weather_model():
    """Test weather prediction model"""
    print("\n" + "="*60)
    print("TESTING WEATHER MODEL")
    print("="*60)

    try:
        # Load model - try best first, then xgboost, then random_forest
        model_path = None
        for name in ["best_weather_model.pkl", "xgboost_weather_model.pkl", "random_forest_weather_model.pkl"]:
            p = Path("data/models") / name
            if p.exists():
                model_path = p
                break

        if model_path is None:
            print("No weather model found! Train one first.")
            return False

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded: {model_path.name}")

        # Load encoders
        with open("data/processed/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        with open("data/processed/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("data/processed/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        n_features = scaler.mean_.shape[0]
        print(f"   Expected features: {n_features}")
        print(f"   Feature names: {feature_names}")

        # Test with sample data
        print("\nTesting with sample weather conditions:")

        samples = [
            {
                "name": "Cold, wet, high humidity (likely blight)",
                "values": {"temperature": 18.0, "humidity": 92.0, "rainfall": 15.0,
                          "wind_speed": 5.0, "dew_point": 17.0}
            },
            {
                "name": "Warm, dry, good ventilation (likely safe)",
                "values": {"temperature": 26.0, "humidity": 65.0, "rainfall": 0.5,
                          "wind_speed": 15.0, "dew_point": 12.0}
            },
            {
                "name": "Moderate, humid (medium risk)",
                "values": {"temperature": 22.0, "humidity": 78.0, "rainfall": 5.0,
                          "wind_speed": 8.0, "dew_point": 18.0}
            }
        ]

        for sample in samples:
            # Build feature vector matching the training feature names
            feature_vec = np.zeros((1, n_features))
            for i, fname in enumerate(feature_names):
                # Try direct match
                if fname in sample["values"]:
                    feature_vec[0, i] = sample["values"][fname]
                # Handle engineered features
                elif fname == "temp_humidity_interaction":
                    feature_vec[0, i] = sample["values"].get("temperature", 0) * sample["values"].get("humidity", 0) / 100
                elif fname == "is_blight_favorable":
                    t = sample["values"].get("temperature", 0)
                    h = sample["values"].get("humidity", 0)
                    feature_vec[0, i] = 1.0 if (15 <= t <= 25 and h >= 85) else 0.0
                elif fname == "temp_dewpoint_diff":
                    feature_vec[0, i] = sample["values"].get("temperature", 0) - sample["values"].get("dew_point", 0)

            scaled = scaler.transform(feature_vec)
            pred = model.predict(scaled)
            label = le.classes_[pred[0]]

            print(f"\n   {sample['name']}")
            print(f"   Prediction: {label}")

        print("\nWeather model working!")
        return True

    except Exception as e:
        print(f"Weather model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_model():
    """Test ViT image classification model"""
    print("\n" + "="*60)
    print("TESTING ViT MODEL")
    print("="*60)

    try:
        import torch
        import timm
        from torchvision import transforms
        from PIL import Image

        device = get_device()

        # Load metadata
        metadata_path = Path("data/processed/metadata.json")
        if not metadata_path.exists():
            print("No metadata.json found! Run preprocessing first.")
            return False

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        class_names = metadata["class_names"]
        idx_to_class = {v: k for k, v in metadata["classes"].items()}

        # Load model
        model_path = Path("data/models/vit_best.pth")
        if not model_path.exists():
            model_path = Path("data/models/vit_final.pth")
        if not model_path.exists():
            print("No trained ViT model found. Run training first.")
            return False

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_name = checkpoint.get('model_name', 'vit_small_patch16_224')

        model = timm.create_model(model_name, pretrained=False, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"Model loaded: {model_name}")
        if 'val_acc' in checkpoint:
            print(f"   Val Accuracy: {checkpoint['val_acc']:.2f}%")

        # Test with sample images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("\nTesting with sample images:")

        val_path = Path("data/processed/val")
        tested = 0
        correct = 0

        if val_path.exists():
            for class_name in class_names:
                class_dir = val_path / class_name
                if not class_dir.exists():
                    continue

                images = (list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) +
                         list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG")))
                test_images = images[:3]  # Test 3 per class

                for img_path in test_images:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = torch.softmax(output, dim=1)
                        pred_idx = probs.argmax(1).item()
                        confidence = probs[0][pred_idx].item() * 100

                    predicted_class = idx_to_class[pred_idx]
                    actual_class = class_name
                    is_correct = predicted_class == actual_class
                    tested += 1
                    correct += int(is_correct)

                    status = "OK" if is_correct else "WRONG"
                    print(f"\n   [{status}] {img_path.name}")
                    print(f"      Actual:    {actual_class}")
                    print(f"      Predicted: {predicted_class} ({confidence:.1f}%)")

        if tested > 0:
            print(f"\nSample accuracy: {correct}/{tested} ({100*correct/tested:.1f}%)")

        print("ViT model working!")
        return True

    except Exception as e:
        print(f"ViT model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import subprocess, sys

    print("=" * 60)
    print("MODEL TESTING SUITE")
    print("=" * 60)
    print("(Running each test in isolated process for MPS stability)\n")

    script = str(Path(__file__).resolve())
    cwd = str(Path(__file__).resolve().parent.parent)

    # Run weather test in subprocess
    r1 = subprocess.run([sys.executable, script, "--weather-only"], cwd=cwd)
    weather_ok = r1.returncode == 0

    # Run ViT test in subprocess
    r2 = subprocess.run([sys.executable, script, "--vit-only"], cwd=cwd)
    vit_ok = r2.returncode == 0

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Weather Model: {'PASSED' if weather_ok else 'FAILED'}")
    print(f"ViT Model:     {'PASSED' if vit_ok else 'FAILED'}")

    if weather_ok and vit_ok:
        print("\nALL TESTS PASSED! Ready to run the dashboard.")
        return 0
    else:
        print("\nSome tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    import sys
    if "--weather-only" in sys.argv:
        exit(0 if test_weather_model() else 1)
    elif "--vit-only" in sys.argv:
        exit(0 if test_vit_model() else 1)
    else:
        exit(main())
