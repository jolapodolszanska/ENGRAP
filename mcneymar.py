import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from statsmodels.stats.contingency_tables import mcnemar
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image

from model_ENGRAP_single_heatmaps import (
    HybridCNN, FocalLoss, CapsuleLayer,
    IMG_SIZE, CLASS_NAMES, DATASET_PATH
)

N_EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_WEIGHTS = [1.0, 7.0, 1.0, 2.0]

def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

train_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

class ResNet50Baseline(pl.LightningModule):
    def __init__(self, n_classes=4):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feat, n_classes)
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0,
                                   weight=torch.tensor(CLASS_WEIGHTS))

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()

        f1_per_class = []
        for c in range(self.hparams.n_classes):
            tp = ((preds == c) & (y == c)).sum().float()
            fp = ((preds == c) & (y != c)).sum().float()
            fn = ((preds != c) & (y == c)).sum().float()
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1_per_class.append(2*p*r / (p + r + 1e-8))
        f1 = torch.stack(f1_per_class).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": sch}

class CapsNetBaseline(pl.LightningModule):
    def __init__(self, n_classes=4):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_feat = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        self.in_capsules = 64
        self.in_dim = 32
        self.out_caps = n_classes  # capsule per class
        self.out_dim = 16

        self.fc_transform = nn.Linear(in_feat, self.in_capsules * self.in_dim)
        self.capsule_layer = CapsuleLayer(
            self.in_capsules, self.in_dim, self.out_caps, self.out_dim, num_routes=3
        )

        self.fc_out = nn.Linear(self.out_caps * self.out_dim, n_classes)
        self.dropout = nn.Dropout(0.5)

        self.criterion = FocalLoss(alpha=1.0, gamma=2.0,
                                   weight=torch.tensor(CLASS_WEIGHTS))

    def forward(self, x):
        f = self.resnet(x)  
        t = self.fc_transform(f)  
        u = t.view(t.size(0), self.in_capsules, self.in_dim) 
        v = self.capsule_layer(u) 
        v_flat = v.view(v.size(0), -1)  
        v_flat = self.dropout(v_flat)
        return self.fc_out(v_flat)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        acc = (preds == y).float().mean()
        f1_per_class = []
        for c in range(self.hparams.n_classes):
            tp = ((preds == c) & (y == c)).sum().float()
            fp = ((preds == c) & (y != c)).sum().float()
            fn = ((preds != c) & (y == c)).sum().float()
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1_per_class.append(2*p*r / (p + r + 1e-8))
        f1 = torch.stack(f1_per_class).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": sch}

def train_baseline(model_cls, model_name, base_dataset, train_idx, val_idx):
    fold_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    train_ds = TransformedSubset(base_dataset, train_idx, train_tf)
    val_ds = TransformedSubset(base_dataset, val_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)

    ckpt_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_from = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_from:
        print(f"  [RESUME] Wznawiam z: {resume_from}")

    model = model_cls()

    best_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_f1", mode="max", patience=10, verbose=True)

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[best_cb, early_stop],
        logger=TensorBoardLogger(save_dir=ckpt_dir, name="tb_logs"),
        enable_progress_bar=True,
        log_every_n_steps=20,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    print(f"\n  Best: {best_cb.best_model_path}")
    print(f"  Time: {(time.time() - fold_start)/60:.1f} min")

    del model, trainer
    torch.cuda.empty_cache()

    return best_cb.best_model_path, val_loader, val_ds

def collect_predictions(checkpoint_path, model_cls, val_loader):
    model = model_cls.load_from_checkpoint(checkpoint_path).to(DEVICE).eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(y.numpy())
    del model
    torch.cuda.empty_cache()
    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_labels))

def mcnemar_test(preds_a, preds_b, labels):
    a_correct = preds_a == labels
    b_correct = preds_b == labels

    n_both_correct = np.sum(a_correct & b_correct)
    n_a_only = np.sum(a_correct & ~b_correct)
    n_b_only = np.sum(~a_correct & b_correct)
    n_both_wrong = np.sum(~a_correct & ~b_correct)

    table = [[n_both_correct, n_a_only],
             [n_b_only, n_both_wrong]]

    result = mcnemar(table, exact=False, correction=True)

    acc_a = a_correct.mean()
    acc_b = b_correct.mean()

    diff = a_correct.astype(int) - b_correct.astype(int)
    cohen_d = diff.mean() / (diff.std() + 1e-8) if diff.std() > 0 else 0.0

    return {
        "chi2": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohen_d": float(cohen_d),
        "acc_a": float(acc_a),
        "acc_b": float(acc_b),
        "n_a_only": int(n_a_only),
        "n_b_only": int(n_b_only),
        "n_both_correct": int(n_both_correct),
        "n_both_wrong": int(n_both_wrong),
    }

def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")

    splits_path = os.path.join("cv_results", "fold_splits.npz")
    if not os.path.exists(splits_path):
        print(f"[ERROR] Brak {splits_path}. Najpierw uruchom compute_5fold_cv.py")
        return
    splits_data = np.load(splits_path)
    train_idx = splits_data["train_0"]
    val_idx = splits_data["val_0"]
    print(f"[INFO] Używam split z fold 1: train={len(train_idx)}, val={len(val_idx)}")

    base_dataset = datasets.ImageFolder(DATASET_PATH, transform=None)

    resnet_ckpt, val_loader, val_ds = train_baseline(
        ResNet50Baseline, "resnet50", base_dataset, train_idx, val_idx
    )

    capsnet_ckpt, _, _ = train_baseline(
        CapsNetBaseline, "capsnet", base_dataset, train_idx, val_idx
    )

    engrap_ckpt = None
    fold1_dir = os.path.join("cv_results", "fold_1")
    if os.path.isdir(fold1_dir):
        bests = [f for f in os.listdir(fold1_dir) if f.startswith("best-") and f.endswith(".ckpt")]
        if bests:
            engrap_ckpt = os.path.join(fold1_dir, bests[0])
    if engrap_ckpt is None:
        print("[ERROR] Brak ENGRAP checkpointu z fold 1.")
        return
    print(f"[INFO] ENGRAP checkpoint (fold 1): {engrap_ckpt}")

    print("\n[INFO] Zbieranie predykcji...")
    eval_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=False)
    engrap_probs, engrap_preds, labels = collect_predictions(engrap_ckpt, HybridCNN, eval_loader)
    resnet_probs, resnet_preds, _ = collect_predictions(resnet_ckpt, ResNet50Baseline, eval_loader)
    capsnet_probs, capsnet_preds, _ = collect_predictions(capsnet_ckpt, CapsNetBaseline, eval_loader)

    metrics = {}
    for name, preds, probs in [
        ("ENGRAP", engrap_preds, engrap_probs),
        ("ResNet50", resnet_preds, resnet_probs),
        ("CapsNet", capsnet_preds, capsnet_probs),
    ]:
        metrics[name] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
            "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        }

    print("\n=== METRYKI ===")
    print(f"{'Model':<10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    for name, m in metrics.items():
        print(f"{name:<10} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    print("\n=== TEST McNEMARA (ENGRAP vs baseline) ===")
    mcnemar_results = []
    for name, preds in [("ResNet50", resnet_preds), ("CapsNet", capsnet_preds)]:
        res = mcnemar_test(engrap_preds, preds, labels)
        res["comparison"] = f"ENGRAP vs {name}"
        mcnemar_results.append(res)

        print(f"\n[ENGRAP vs {name}]")
        print(f"  ENGRAP accuracy:    {res['acc_a']:.4f}")
        print(f"  {name:<10} accuracy: {res['acc_b']:.4f}")
        print(f"  Contingency table:")
        print(f"    both correct: {res['n_both_correct']}")
        print(f"    ENGRAP only:  {res['n_a_only']}")
        print(f"    {name} only:    {res['n_b_only']}")
        print(f"    both wrong:   {res['n_both_wrong']}")
        print(f"  McNemar χ² = {res['chi2']:.2f}")
        print(f"  p-value    = {res['p_value']:.4e}")
        print(f"  Cohen's d  = {res['cohen_d']:.3f}")

    import pandas as pd
    pd.DataFrame(mcnemar_results).to_csv(
        os.path.join(OUTPUT_DIR, "mcnemar_results.csv"), index=False
    )
    pd.DataFrame(metrics).T.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))

    print("\n\n=== GOTOWY TEKST DO TABELI 3 ===\n")
    for res in mcnemar_results:
        sig = "***" if res["p_value"] < 0.001 else ("**" if res["p_value"] < 0.01 else
              ("*" if res["p_value"] < 0.05 else "ns"))
        d_size = "Very large" if abs(res["cohen_d"]) >= 1.2 else (
                 "Large" if abs(res["cohen_d"]) >= 0.8 else (
                 "Medium" if abs(res["cohen_d"]) >= 0.5 else (
                 "Small" if abs(res["cohen_d"]) >= 0.2 else "Negligible")))
        p_str = f"< 0.001" if res["p_value"] < 0.001 else f"{res['p_value']:.4f}"
        print(f"{res['comparison']:<25} χ²={res['chi2']:>7.2f}  "
              f"p={p_str:<10}  d={res['cohen_d']:+.3f}  ({d_size})  {sig}")


if __name__ == "__main__":
    mp.freeze_support()
    main()