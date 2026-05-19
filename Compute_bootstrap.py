import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model_ENGRAP_single_heatmaps import HybridCNN, CustomDataModule, DATASET_PATH

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_BOOTSTRAP = 1000
    SEED = 42

    ckpts = sorted(glob.glob(os.path.join("checkpoints", "captrad-*.ckpt")))
    best, best_f1 = None, -1.0
    for p in ckpts:
        m = re.search(r"val_f1=([\d]+\.[\d]+)", p)
        if m and float(m.group(1)) > best_f1:
            best_f1 = float(m.group(1))
            best = p
    print(f"Checkpoint: {best}")

    model = HybridCNN.load_from_checkpoint(best).to(DEVICE).eval()

    dm = CustomDataModule(DATASET_PATH)
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    n = len(all_labels)
    print(f"Total val samples: {n}\n")

    rng = np.random.RandomState(SEED)
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-score": [],
        "AUC": [],
    }

    print(f"Running {N_BOOTSTRAP} bootstrap resamples...")
    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        y_true = all_labels[idx]
        y_pred = all_preds[idx]
        y_prob = all_probs[idx]

        if len(np.unique(y_true)) < 4:
            continue

        metrics["Accuracy"].append(accuracy_score(y_true, y_pred))
        metrics["Precision"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["Recall"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["F1-score"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            metrics["AUC"].append(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            pass

    point = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "Recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "F1-score": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "AUC": roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"),
    }

    print(f"\n{'Metric':<12} {'Point':>8} {'CI lower':>10} {'CI upper':>10}")
    print("-" * 44)
    for name in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
        vals = np.array(metrics[name])
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        print(f"{name:<12} {point[name]:>8.4f} {lo:>10.4f} {hi:>10.4f}")

    print(f"\nGotowy tekst do manuskryptu:")
    print(f"Bootstrap confidence intervals (95% CI, n = {N_BOOTSTRAP} resamples) for ENGRAP performance:")
    for name in ["Accuracy", "Precision", "Recall", "F1-score"]:
        vals = np.array(metrics[name])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {name} {point[name]:.3f} [{lo:.3f}, {hi:.3f}]")