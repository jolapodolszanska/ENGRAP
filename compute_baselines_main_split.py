import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model_ENGRAP_single_heatmaps import (
    HybridCNN, FocalLoss, CapsuleLayer, CustomDataModule,
    DATASET_PATH, BATCH_SIZE
)


BATCH_SIZE_TRAIN = 64
NUM_WORKERS = 4
N_EPOCHS = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "baselines_main"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_WEIGHTS = [1.0, 7.0, 1.0, 2.0]


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        in_feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.in_capsules = 64
        self.in_dim = 32
        self.out_caps = n_classes
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

def train_model(model_cls, model_name, dm):
    fold_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training {model_name} on MAIN split (val_split=0.2, seed=42)")
    print(f"{'=' * 60}")
    print(f"  Train: {len(dm.train_ds)}  |  Val: {len(dm.val_ds)}")

    train_loader = DataLoader(dm.train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(dm.val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=False,
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
        monitor="val_f1", mode="max", save_top_k=1, save_last=True,
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

    return best_cb.best_model_path, val_loader

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


def compute_metrics(preds, probs, labels):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
    }

def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Using CustomDataModule with val_split=0.2, seed=42")
    print(f"[INFO] Dataset: {DATASET_PATH}\n")

    dm = CustomDataModule(DATASET_PATH)
    dm.setup("fit")

    resnet_ckpt, val_loader = train_model(ResNet50Baseline, "resnet50", dm)

    capsnet_ckpt, _ = train_model(CapsNetBaseline, "capsnet", dm)

    print("\n[INFO] Ewaluacja na walidacji...")
    eval_loader = DataLoader(dm.val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=False)

    resnet_probs, resnet_preds, labels = collect_predictions(resnet_ckpt, ResNet50Baseline, eval_loader)
    capsnet_probs, capsnet_preds, _ = collect_predictions(capsnet_ckpt, CapsNetBaseline, eval_loader)

    import glob, re
    engrap_ckpts = sorted(glob.glob(os.path.join("checkpoints", "captrad-*.ckpt")))
    best_engrap, best_f1 = None, -1.0
    for p in engrap_ckpts:
        m = re.search(r"val_f1=([\d]+\.[\d]+)", p)
        if m and float(m.group(1)) > best_f1:
            best_f1 = float(m.group(1))
            best_engrap = p
    print(f"[INFO] ENGRAP checkpoint: {best_engrap}")
    engrap_probs, engrap_preds, _ = collect_predictions(best_engrap, HybridCNN, eval_loader)

    print("\n=== METRYKI (główna walidacja, n={}) ===".format(len(labels)))
    metrics = {
        "ENGRAP": compute_metrics(engrap_preds, engrap_probs, labels),
        "ResNet50": compute_metrics(resnet_preds, resnet_probs, labels),
        "CapsNet": compute_metrics(capsnet_preds, capsnet_probs, labels),
    }
    print(f"{'Model':<10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    for name, m in metrics.items():
        print(f"{name:<10} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    from statsmodels.stats.contingency_tables import mcnemar

    def mcnemar_pair(preds_a, preds_b, labels, name_a, name_b):
        a_ok = preds_a == labels
        b_ok = preds_b == labels
        table = [[(a_ok & b_ok).sum(), (a_ok & ~b_ok).sum()],
                 [(~a_ok & b_ok).sum(), (~a_ok & ~b_ok).sum()]]
        res = mcnemar(table, exact=False, correction=True)
        diff = a_ok.astype(int) - b_ok.astype(int)
        d = diff.mean() / (diff.std() + 1e-8) if diff.std() > 0 else 0.0
        print(f"\n  {name_a} vs {name_b}:")
        print(f"    χ²={res.statistic:.2f}, p={res.pvalue:.4e}, d={d:.3f}")
        return {"chi2": float(res.statistic), "p_value": float(res.pvalue),
                "cohen_d": float(d), "comparison": f"{name_a} vs {name_b}"}

    print("\n=== TEST McNEMARA (n={}) ===".format(len(labels)))
    mcnemar_res = []
    mcnemar_res.append(mcnemar_pair(engrap_preds, resnet_preds, labels, "ENGRAP", "ResNet50"))
    mcnemar_res.append(mcnemar_pair(engrap_preds, capsnet_preds, labels, "ENGRAP", "CapsNet"))

    pd.DataFrame(metrics).T.to_csv(os.path.join(OUTPUT_DIR, "metrics_main_split.csv"))
    pd.DataFrame(mcnemar_res).to_csv(
        os.path.join(OUTPUT_DIR, "mcnemar_main_split.csv"), index=False)

    print("\n\n=== GOTOWY TEKST DO TABELI 2 (własne baseliny + ENGRAP) ===\n")
    for name in ["ResNet50", "CapsNet", "ENGRAP"]:
        m = metrics[name]
        print(f"{name:<15} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
              f"{m['recall']:.3f} & {m['f1']:.3f} & {m['auc']:.3f} \\\\")

    print("\n=== GOTOWY TEKST DO TABELI 3 ===\n")
    for res in mcnemar_res:
        p = res["p_value"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        d_size = ("Very large" if abs(res["cohen_d"]) >= 1.2 else
                  "Large" if abs(res["cohen_d"]) >= 0.8 else
                  "Medium" if abs(res["cohen_d"]) >= 0.5 else
                  "Small" if abs(res["cohen_d"]) >= 0.2 else
                  "Negligible")
        p_str = "< 0.001" if p < 0.001 else f"{p:.4f}"
        print(f"{res['comparison']:<25} & {res['chi2']:.2f} & {p_str}{sig} & "
              f"{res['cohen_d']:.3f} & {d_size} \\\\")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()