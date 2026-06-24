from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cifar10_data import CIFAR10_CLASSES, CIFAR10Dataset
from model import build_model


def load_image(path: str | Path) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((32, 32))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(32, 32, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    return (tensor - CIFAR10Dataset.mean) / CIFAR10Dataset.std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a CIFAR-10 class with a trained ResNet.")
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument("--model-path", default="models/resnet20_cifar10.pt", help="Checkpoint path.")
    parser.add_argument("--model", default=None, help="Override model name stored in checkpoint.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model_name = args.model or checkpoint.get("model", "resnet20")
    class_names = checkpoint.get("class_names", CIFAR10_CLASSES)

    model = build_model(model_name, num_classes=len(class_names))
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    image = load_image(args.image).unsqueeze(0)
    probs = torch.softmax(model(image), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    print(f"Prediction: {class_names[index]} ({confidence.item():.2%})")


if __name__ == "__main__":
    main()


