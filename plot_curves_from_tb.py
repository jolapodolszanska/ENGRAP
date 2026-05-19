import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_DIR   = r"F:\ENGRAP\tb_logs\engrap"
OUT_PATH = r"F:\ENGRAP\engrap_training_curves.png"


versions = sorted(glob.glob(os.path.join(TB_DIR, "version_*")),
                  key=lambda p: int(p.split("version_")[-1]))

best_ver, best_count = None, 0
for ver in versions:
    events = glob.glob(os.path.join(ver, "events.out.tfevents.*"))
    size = sum(os.path.getsize(e) for e in events)
    if size > best_count:
        best_count = size
        best_ver = ver

print(f"Używam wersji: {best_ver}")

ea = EventAccumulator(best_ver)
ea.Reload()
print("Dostępne tagi:", ea.Tags().get("scalars", []))

def get_series(tag):
    try:
        events = ea.Scalars(tag)
        steps  = [e.step for e in events]
        vals   = [e.value for e in events]
        return np.array(steps), np.array(vals)
    except KeyError:
        return None, None

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=130)

ax = axes[0]
for tag, label, color, ls in [
    ("train_loss_epoch", "Train loss", "#4e79a7", "-"),
    ("train_loss",       "Train loss", "#4e79a7", "-"),
    ("val_loss",         "Val loss",   "#e15759", "--"),
]:
    steps, vals = get_series(tag)
    if steps is not None:
        ax.plot(steps, vals, label=label, color=color, linewidth=2, linestyle=ls)
        break  

steps, vals = get_series("val_loss")
if steps is not None:
    ax.plot(steps, vals, label="Val loss", color="#e15759", linewidth=2, linestyle="--")

ax.set_xlabel("Epoch / Step", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("Training and Validation Loss", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Accuracy ──
ax2 = axes[1]
steps, vals = get_series("train_acc")
if steps is not None:
    ax2.plot(steps, vals, label="Train accuracy", color="#4e79a7", linewidth=2)
steps, vals = get_series("val_acc")
if steps is not None:
    ax2.plot(steps, vals, label="Val accuracy", color="#e15759",
             linewidth=2, linestyle="--")

ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Accuracy", fontsize=10)
ax2.set_title("Training and Validation Accuracy", fontsize=11)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {OUT_PATH}")
