import os
import time
import numpy as np
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
    HybridCNN, FocalLoss, CustomDataModule, DATASET_PATH
)

BATCH_SIZE_TRAIN = 64
NUM_WORKERS = 4
N_EPOCHS = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_WEIGHTS = [1.0, 7.0, 1.0, 2.0]


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Wariant ablation: ResNet50 + Transformer (bez Capsule)
# ============================================================================

class ResNetTransformerBaseline(pl.LightningModule):
    """
    ResNet50 backbone -> linear projection to tokens -> [CLS] + Transformer encoder
    -> klasyfikacja przez CLS. Bez CapsuleLayer, bez late fusion (zgodnie
    z duchem ablation - tylko Transformer dodany do ResNet).
    
    Architektura analogiczna do ENGRAP, ale Capsule -> Linear (token sequence).
    """
    def __init__(self, n_classes=4):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_feat = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Projekcja do tokenów (10 tokenów x 16-dim, analogicznie do output CapsuleLayer)
        self.n_tokens = 10
        self.d_model = 16
        self.fc_transform = nn.Linear(in_feat, self.n_tokens * self.d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Transformer encoder (L=2, H=4, d=16) - identyczny jak w ENGRAP
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=64,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Klasyfikator z CLS
        self.fc_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

        self.criterion = FocalLoss(alpha=1.0, gamma=2.0,
                                   weight=torch.tensor(CLASS_WEIGHTS))

    def forward(self, x):
        B = x.size(0)
        f = self.resnet(x)  # [B, 2048]
        t = self.fc_transform(f)  # [B, n_tokens * d_model]
        tokens = t.view(B, self.n_tokens, self.d_model)  # [B, 10, 16]

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, 16]
        seq = torch.cat([cls, tokens], dim=1)  # [B, 11, 16]

        y = self.transformer(seq)  # [B, 11, 16]
        cls_out = y[:, 0]  # [B, 16]
        return self.fc_out(cls_out)

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


# ============================================================================
# Trening
# ============================================================================

def train_model(model_cls, model_name, dm):
    fold_start = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
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

    return best_cb.best_model_path


# ============================================================================
# Ewaluacja
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Using CustomDataModule with val_split=0.2, seed=42")

    dm = CustomDataModule(DATASET_PATH)
    dm.setup("fit")

    # 1. Trening ResNet50 + Transformer (jedyny nowy trening)
    rt_ckpt = train_model(ResNetTransformerBaseline, "resnet50_transformer", dm)

    # 2. Zbieranie checkpointów z istniejących treningów
    import glob, re

    # ResNet50 (baseline) - z baselines_main/resnet50/
    resnet_ckpts = glob.glob(os.path.join("baselines_main", "resnet50", "best-*.ckpt"))
    resnet_ckpt = max(resnet_ckpts, key=os.path.getmtime) if resnet_ckpts else None
    print(f"[INFO] ResNet50 ckpt: {resnet_ckpt}")

    # ResNet50 + Capsules - to baseline CapsNet z baselines_main/capsnet/
    capsnet_ckpts = glob.glob(os.path.join("baselines_main", "capsnet", "best-*.ckpt"))
    capsnet_ckpt = max(capsnet_ckpts, key=os.path.getmtime) if capsnet_ckpts else None
    print(f"[INFO] ResNet50+Capsules ckpt (=CapsNet baseline): {capsnet_ckpt}")

    # ENGRAP - z checkpoints/captrad-*.ckpt
    engrap_ckpts = sorted(glob.glob(os.path.join("checkpoints", "captrad-*.ckpt")))
    best_engrap, best_f1 = None, -1.0
    for p in engrap_ckpts:
        m = re.search(r"val_f1=([\d]+\.[\d]+)", p)
        if m and float(m.group(1)) > best_f1:
            best_f1 = float(m.group(1))
            best_engrap = p
    print(f"[INFO] ENGRAP ckpt: {best_engrap}")

    if not (resnet_ckpt and capsnet_ckpt and best_engrap and rt_ckpt):
        print("[ERROR] Brakuje któregoś z checkpointów.")
        return

    # 3. Importujemy klasy modeli (lokalnie tu, żeby uniknąć cykli)
    from compute_baselines_main_split import ResNet50Baseline, CapsNetBaseline

    eval_loader = DataLoader(dm.val_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=False)

    print("\n[INFO] Ewaluacja wszystkich 4 wariantów...")
    metrics = {}

    print("  -> ResNet50")
    probs, preds, labels = collect_predictions(resnet_ckpt, ResNet50Baseline, eval_loader)
    metrics["ResNet50"] = compute_metrics(preds, probs, labels)

    print("  -> ResNet50 + Capsules")
    probs, preds, _ = collect_predictions(capsnet_ckpt, CapsNetBaseline, eval_loader)
    metrics["ResNet50 + Capsules"] = compute_metrics(preds, probs, labels)

    print("  -> ResNet50 + Transformer")
    probs, preds, _ = collect_predictions(rt_ckpt, ResNetTransformerBaseline, eval_loader)
    metrics["ResNet50 + Transformer"] = compute_metrics(preds, probs, labels)

    print("  -> ENGRAP (full)")
    probs, preds, _ = collect_predictions(best_engrap, HybridCNN, eval_loader)
    metrics["ENGRAP"] = compute_metrics(preds, probs, labels)

    # 4. Print + zapis
    print("\n=== ABLATION RESULTS (główna walidacja, n={}) ===".format(len(labels)))
    print(f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    for name, m in metrics.items():
        print(f"{name:<28} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    import pandas as pd
    pd.DataFrame(metrics).T.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"))

    # 5. Gotowy tekst LaTeX do Tabeli 4
    print("\n\n=== GOTOWY TEKST DO TABELI 4 (ablation, LaTeX) ===\n")
    for name in ["ResNet50", "ResNet50 + Capsules", "ResNet50 + Transformer", "ENGRAP"]:
        m = metrics[name]
        bold = "\\textbf{" if name == "ENGRAP" else ""
        end = "}" if name == "ENGRAP" else ""
        print(f"{bold}{name}{end} & {bold}{m['accuracy']:.3f}{end} & "
              f"{bold}{m['precision']:.3f}{end} & {bold}{m['recall']:.3f}{end} & "
              f"{bold}{m['f1']:.3f}{end} & {bold}{m['auc']:.3f}{end} \\\\")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()