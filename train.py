from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import json
import os

import argparse

from dataset import CIFAR10Dataset, load_data
from dataset import train_transform, test_transform
from model import FullModel
from utils import CrossEntropyLoss, StochasticGradientDescent

def train():
    # PARSER
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--mlp_mult", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--optimizer",choices=["SGD", "AdamW"],default="AdamW",type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--no_augment", action="store_true")

    parser.add_argument("--out", default=None, type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # DATA

    train_images, train_labels, test_images, test_labels = load_data()
    train_transform = None if args.no_augment else train_transform
    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=train_transform)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    C, H, W = train_loader.dataset[0][0].shape

    # MODEL INIT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    def gpu_prep(x):
        x = x.to(device, non_blocking=True).float().div_(255)
        return (x - mean) / std
    
    model = FullModel(H=H, C=C, P=args.patch_size, D=args.hidden_dim, num_heads=args.num_heads, mlp_mult=args.mlp_mult, depth=args.depth).to(device)

    criterion = CrossEntropyLoss()

    if args.optimizer == "SGD":
        optimizer = StochasticGradientDescent(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs]
    )

    history = {"train_loss": [], "train_acc": [], "test_acc": [], "lr": []}

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = gpu_prep(images)                       
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model.forward(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # VALIDATION / TEST
        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = gpu_prep(images)                     
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Test Acc: {test_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step()
        if args.out is not None:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            # History + Args (wie bisher)
            with open(args.out, "w") as f:
                json.dump({"args": vars(args), "history": history}, f, indent=2)
            print(f"Gespeichert: {args.out}")

            # Modell: neben der JSON, mit Endung .pt
            ckpt_path = args.out + ".pt"
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "model_kwargs": dict(
                    H=H, C=C, P=args.patch_size, D=args.hidden_dim,
                    num_heads=args.num_heads, mlp_mult=args.mlp_mult, depth=args.depth,
                ),
                "epoch": args.epochs,
                "test_acc": history["test_acc"][-1],
            }, ckpt_path)
            print(f"Modell gespeichert: {ckpt_path}")


if __name__=="__main__":
    train()