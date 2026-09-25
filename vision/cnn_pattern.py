"""P2C-02 ResNet18 微调识别 K 线形态（5 类）。

数据：合成形态（主）+ 可选人工标注；Mac CPU 训练 5-10 epoch。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

from models.common import MODELS_DIR, RESULTS_DIR, get_device, save_json
from vision.render import PATTERNS, IMG_SIZE

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class PatternDataset(Dataset):
    def __init__(self, images: list[Image.Image], labels: list[int]):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return TRANSFORM(self.images[i]), self.labels[i]


def build_model(n_classes: int = len(PATTERNS)) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def train_pattern_cnn(images, labels, epochs: int = 8, batch_size: int = 32,
                      lr: float = 1e-3, model_path: Path | None = None) -> nn.Module:
    device = get_device()
    ds = PatternDataset(images, labels)
    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), desc="CNN 形态识别训练"):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                n += len(y)
        print(f"epoch {epoch}: loss={total/len(train_ds):.4f} val_acc={correct/n:.4f}")

    path = model_path or MODELS_DIR / "pattern_cnn.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": PATTERNS}, path)
    print(f"CNN 模型已保存: {path}")
    return model


def evaluate_pattern_cnn(model, images, labels) -> dict:
    device = get_device()
    ds = PatternDataset(images, labels)
    dl = DataLoader(ds, batch_size=64)
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y_true.extend(y.tolist())
            y_pred.extend(model(x).argmax(1).cpu().tolist())
    report = classification_report(y_true, y_pred, target_names=PATTERNS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    result = {"accuracy": round(float(acc), 4), "confusion_matrix": cm,
              "per_class": {p: {k: round(v, 4) for k, v in report[p].items() if isinstance(v, (int, float))}
                            for p in PATTERNS}}
    save_json(result, RESULTS_DIR / "pattern_cnn_metrics.json")
    print(f"CNN 测试准确率: {acc:.4f}")
    return result


def predict_pattern_probs(model, images: list[Image.Image]) -> np.ndarray:
    """K线图 → 形态概率向量 (N, 5)。"""
    device = get_device()
    ds = PatternDataset(images, [0] * len(images))
    dl = DataLoader(ds, batch_size=64)
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs)


def load_pattern_cnn(model_path: Path | None = None) -> nn.Module:
    path = model_path or MODELS_DIR / "pattern_cnn.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model


if __name__ == "__main__":
    from vision.render import build_synthetic_dataset
    print("生成合成形态数据集...")
    imgs, labs = build_synthetic_dataset(n_per_class=150)
    model = train_pattern_cnn(imgs, labs, epochs=5)
    evaluate_pattern_cnn(model, imgs[len(imgs) // 5:], labs[len(labs) // 5:])
