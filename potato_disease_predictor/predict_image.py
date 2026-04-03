#!/usr/bin/env python3
"""Standalone ViT inference script - called as subprocess to avoid MPS+pickle segfault"""

import sys
import json
import torch
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path


def predict(image_path):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model_path = Path("data/models/vit_best.pth")
    if not model_path.exists():
        model_path = Path("data/models/vit_final.pth")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = timm.create_model(checkpoint.get("model_name", "vit_small_patch16_224"),
                              pretrained=False, num_classes=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    meta_path = Path("data/processed/metadata.json")
    idx_to_class = {0: "Potato___Early_blight", 1: "Potato___Late_blight", 2: "Potato___healthy"}
    if meta_path.exists():
        with open(meta_path) as f:
            idx_to_class = {v: k for k, v in json.load(f)["classes"].items()}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]

    result = {}
    for i, cls in idx_to_class.items():
        result[cls] = round(probs[i].item() * 100, 2)

    pred_idx = probs.argmax().item()
    result["_prediction"] = idx_to_class[pred_idx]
    result["_confidence"] = round(probs[pred_idx].item() * 100, 2)

    print(json.dumps(result))


if __name__ == "__main__":
    predict(sys.argv[1])
