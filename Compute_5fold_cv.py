import os
import json
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model_ENGRAP_single_heatmaps import (
    HybridCNN, IMG_SIZE, CLASS_NAMES, DATASET_PATH
)

N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "cv_results"
STATE_FILE = os.path.join(OUTPUT_DIR, "cv_state.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "cv_results.csv")
SPLITS_FILE = os.path.join(OUTPUT_DIR, "fold_splits.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

train_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"completed_folds": [], "results": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def save_csv(results):
    if not results:
        return
    import pandas as pd
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

def get_or_create_splits():
    """Stratyfikowane splity - takie same przy każdym uruchomieniu."""
    base_dataset = datasets.ImageFolder(DATASET_PATH, transform=None)
    labels = np.array([s[1] for s in base_dataset.samples])

    valid_indices = np.array([
        i for i, (path, _) in enumerate(base_dataset.samples)
        if not os.path.basename(path).startswith(("aug_", "topup_"))
    ])
    valid_labels = labels[valid_indices]

    print(f"[INFO] Total samples: {len(valid_indices)}")
    print(f"[INFO] Class distribution: {np.bincount(valid_labels).tolist()}")

    if os.path.exists(SPLITS_FILE):
        print(f"[INFO] Wczytuję istniejące splity: {SPLITS_FILE}")
        data = np.load(SPLITS_FILE, allow_pickle=True)
        splits = [(data[f"train_{i}"], data[f"val_{i}"]) for i in range(N_FOLDS)]
    else:
        print(f"[INFO] Generuję nowe splity (seed={SEED})")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = []
        save_dict = {}
        for i, (train_pos, val_pos) in enumerate(skf.split(valid_indices, valid_labels)):
            train_idx = valid_indices[train_pos]
            val_idx = valid_indices[val_pos]
            splits.append((train_idx, val_idx))
            save_dict[f"train_{i}"] = train_idx
            save_dict[f"val_{i}"] = val_idx
        np.savez(SPLITS_FILE, **save_dict)
        print(f"[INFO] Splity zapisane: {SPLITS_FILE}")

    return base_dataset, valid_labels, splits

def train_fold(fold_idx, base_dataset, train_idx, val_idx):
    fold_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx}/{N_FOLDS}")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    train_ds = TransformedSubset(base_dataset, train_idx, train_tf)
    val_ds = TransformedSubset(base_dataset, val_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)

    ckpt_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")

    resume_from = last_ckpt if os.path.exists(last_ckpt) else None
    if resume_from:
        print(f"  [RESUME] Wznawiam z: {resume_from}")
    else:
        print(f"  [START] Świeży trening")

    model = HybridCNN()

    best_ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,   
    )

    early_stop = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=10,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        callbacks=[best_ckpt_cb, early_stop],
        logger=TensorBoardLogger(save_dir=ckpt_dir, name="tb_logs"),
        enable_progress_bar=True,
        log_every_n_steps=20,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)

    best_path = best_ckpt_cb.best_model_path
    print(f"\n  Best checkpoint: {best_path}")
    best_model = HybridCNN.load_from_checkpoint(best_path).to(DEVICE).eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = best_model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "fold": fold_idx,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")),
        "fold_time_min": round((time.time() - fold_start) / 60.0, 2),
        "best_checkpoint": best_path,
    }

    print(f"\n  FOLD {fold_idx} RESULTS:")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"    {k:<15} {metrics[k]:.4f}")
    print(f"    {'time (min)':<15} {metrics['fold_time_min']:.1f}")

    del model, best_model, trainer
    torch.cuda.empty_cache()

    return metrics

def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Dataset: {DATASET_PATH}")
    print(f"[INFO] Folds: {N_FOLDS}, Epochs/fold: {N_EPOCHS}, Batch: {BATCH_SIZE}\n")

    base_dataset, valid_labels, splits = get_or_create_splits()

    state = load_state()
    completed = set(state["completed_folds"])
    if completed:
        print(f"\n[RESUME] Już ukończone foldy: {sorted(completed)}")
        print(f"[RESUME] Pozostało: {sorted(set(range(1, N_FOLDS+1)) - completed)}")
    else:
        print(f"\n[NEW] Świeże uruchomienie\n")

    total_start = time.time()
    for fold_idx in range(1, N_FOLDS + 1):
        if fold_idx in completed:
            print(f"\n[SKIP] Fold {fold_idx} już ukończony, pomijam.")
            continue

        train_idx, val_idx = splits[fold_idx - 1]

        try:
            metrics = train_fold(fold_idx, base_dataset, train_idx, val_idx)

            state["results"].append(metrics)
            state["completed_folds"].append(fold_idx)
            save_state(state)
            save_csv(state["results"])
            print(f"\n[SAVED] Wyniki zapisane: {RESULTS_CSV}")

        except KeyboardInterrupt:
            print(f"\n\n[INTERRUPTED] Przerwane przez użytkownika.")
            print(f"  Stan: {len(state['completed_folds'])}/{N_FOLDS} foldów ukończonych.")
            print(f"  Aby kontynuować - uruchom skrypt ponownie.")
            return

    total_time = (time.time() - total_start) / 60.0
    print(f"\n\n{'=' * 60}")
    print(f"CV COMPLETE in {total_time:.1f} min")
    print(f"{'=' * 60}\n")

    import pandas as pd
    df = pd.DataFrame(state["results"]).sort_values("fold")
    print("Per-fold results:")
    print(df[["fold", "accuracy", "precision", "recall", "f1", "auc"]].to_string(index=False))

    print(f"\nMean ± std across {N_FOLDS} folds:")
    print(f"  Accuracy  {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  Precision {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
    print(f"  Recall    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
    print(f"  F1-score  {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    print(f"  AUC       {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")

    print(f"\n>>> GOTOWY TEKST DO MANUSKRYPTU (sekcja 4.2.1): <<<\n")
    print(f"Five-fold cross-validation confirmed the robustness of the ENGRAP model, "
          f"yielding the following mean performance metrics with standard deviations: "
          f"Accuracy {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}, "
          f"Precision {df['precision'].mean():.3f} ± {df['precision'].std():.3f}, "
          f"Recall {df['recall'].mean():.3f} ± {df['recall'].std():.3f}, "
          f"F1-score {df['f1'].mean():.3f} ± {df['f1'].std():.3f}, and "
          f"AUC {df['auc'].mean():.3f} ± {df['auc'].std():.3f}.")

if __name__ == "__main__":
    main()