from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar10_data import CIFAR10_CLASSES, CIFAR10Dataset
from model import MODEL_BUILDERS, build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet on CIFAR-10.")
    parser.add_argument("--data-dir", default="data/cifar10", help="CIFAR-10 data directory.")
    parser.add_argument("--model", default="resnet20", choices=sorted(MODEL_BUILDERS), help="Model variant.")
    parser.add_argument("--model-path", default="models/resnet20_cifar10.pt", help="Where to save weights.")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use a smaller train set.")
    parser.add_argument("--limit-test", type=int, default=None, help="Use a smaller test set.")
    parser.add_argument("--no-augment", action="store_true", help="Disable random crop and flip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")

    train_dataset = CIFAR10Dataset(
        args.data_dir,
        train=True,
        augment=not args.no_augment,
        limit=args.limit_train,
    )
    test_dataset = CIFAR10Dataset(args.data_dir, train=False, augment=False, limit=args.limit_test)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1,
    )

    best_acc = 0.0
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": args.model,
                    "num_classes": len(CIFAR10_CLASSES),
                    "class_names": CIFAR10_CLASSES,
                    "state_dict": model.state_dict(),
                    "accuracy": best_acc,
                },
                model_path,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={scheduler.get_last_lr()[0]:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.2%} | "
            f"best={best_acc:.2%}"
        )

    print(f"Saved best checkpoint to {model_path}")


if __name__ == "__main__":
    main()
