import os, re, glob
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model_ENGRAP_single_heatmaps import HybridCNN, CustomDataModule, DATASET_PATH

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Najlepszy checkpoint
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

    # Zbierz prob i labele
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    print(f"Total val samples: {len(all_labels)}")

    auc_macro = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")

    print(f"\nAUC macro (OvR):    {auc_macro:.4f}")
    print(f"AUC weighted (OvR): {auc_weighted:.4f}")

    print("\nPer-class AUC:")
    for i, name in enumerate(["NonDemented", "VeryMild", "Mild", "Moderate"]):
        y_bin = (all_labels == i).astype(int)
        auc_i = roc_auc_score(y_bin, all_probs[:, i])
        print(f"  {name:<13} AUC = {auc_i:.4f}")