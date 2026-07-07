"""Train the ViT on a surface-defect dataset.

Examples
--------
From-scratch ViT::

    python -m src.train --data-dir /path/to/surface_crack --epochs 15

Fine-tune a pretrained ViT from timm (stronger, needs `pip install timm`)::

    python -m src.train --data-dir /path/to/surface_crack --pretrained --epochs 5
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .vit import vit_tiny
from .data import build_loaders


def build_model(pretrained, n_classes, img_size):
    if pretrained:
        import timm
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                  num_classes=n_classes)
        return model, "timm/vit_tiny_patch16_224 (pretrained)"
    return vit_tiny(img_size=img_size, n_classes=n_classes), "from-scratch vit_tiny"


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = total = 0
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.view(-1), pred.view(-1)):
            conf[t.long(), p.long()] += 1
    return correct / max(total, 1), conf


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, classes = build_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        val_split=args.val_split, num_workers=args.num_workers,
    )
    n_classes = len(classes)
    print(f"classes: {classes}")

    model, tag = build_model(args.pretrained, n_classes, args.img_size)
    model.to(device)
    print(f"model: {tag}  params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        val_acc, conf = evaluate(model, val_loader, device, n_classes)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:02d}  train_loss {train_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes,
                        "val_acc": val_acc, "tag": tag},
                       out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(
        {"tag": tag, "classes": classes, "best_val_acc": best_acc,
         "history": history, "confusion_matrix": conf.tolist()}, indent=2))
    print(f"best val_acc: {best_acc:.4f}  ->  {out_dir/'best.pt'}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pretrained", action="store_true",
                   help="fine-tune a pretrained timm ViT instead of training from scratch")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
