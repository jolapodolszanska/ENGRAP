import os
import sys
import json
import random
from collections import defaultdict
from typing import Callable, Tuple, List, Dict
import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from scipy.stats import pearsonr, wilcoxon
from skimage.segmentation import slic
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_ENGRAP_single_heatmaps import (
    HybridCNN, CustomDataModule, GradCAM,
    IMG_SIZE, USE_IMAGENET_NORM, CLASS_NAMES, DATASET_PATH
)

N_SAMPLES_PER_CLASS = 20          
INSERTION_DELETION_STEPS = 50     
LIME_N_SAMPLES = 1000
LIME_N_SEGMENTS = 100
SHAP_N_BACKGROUND = 30
RISE_N_MASKS = 1024
RISE_MASK_RES = 16
RISE_KEEP_PROB = 0.5
SCORECAM_BATCH = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

OUTPUT_DIR = "xai_benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm(img_tensor: torch.Tensor) -> np.ndarray:
    x = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if USE_IMAGENET_NORM:
        x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0.0, 1.0)

def norm_np(img_np: np.ndarray) -> torch.Tensor:
    if USE_IMAGENET_NORM:
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_np.transpose(2, 0, 1).astype(np.float32))

def predict_proba_batch(model: nn.Module, imgs_np: np.ndarray,
                        batch_size: int = 32) -> np.ndarray:
    """imgs_np: [N,H,W,3] w [0,1] (bez normalizacji). Zwraca [N,K] prawdopodobieństw."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs_np), batch_size):
            batch = imgs_np[i:i+batch_size]
            tensors = torch.stack([norm_np(im) for im in batch]).to(DEVICE)
            logits = model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


def normalize_attr(attr: np.ndarray) -> np.ndarray:
    a = np.asarray(attr, dtype=np.float32)
    a = np.abs(a)  
    if a.max() - a.min() < 1e-12:
        return np.zeros_like(a)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def build_stratified_subset(data_dir: str, n_per_class: int = N_SAMPLES_PER_CLASS,
                            split: str = "val") -> List[Tuple[torch.Tensor, int]]:
    eval_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.1)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        split_dir = data_dir
    full = datasets.ImageFolder(split_dir, transform=eval_tf)

    valid_indices = [
        i for i, (path, _) in enumerate(full.samples)
        if not os.path.basename(path).startswith(("aug_", "topup_"))
    ]

    per_class = defaultdict(list)
    for i in valid_indices:
        _, label = full.samples[i]
        per_class[label].append(i)

    rng = np.random.RandomState(SEED)
    selected = []
    for cls in sorted(per_class.keys()):
        idxs = per_class[cls]
        rng.shuffle(idxs)
        chosen = idxs[:n_per_class]
        for i in chosen:
            img, label = full[i]
            selected.append((img, label))

    print(f"[INFO] Wybrano {len(selected)} obrazów ({n_per_class}/klasę)")
    return selected

def compute_gradcam(model: nn.Module, img_tensor: torch.Tensor,
                    target_class: int) -> np.ndarray:
    cam_engine = GradCAM(model, model.resnet.layer4)
    img_dev = img_tensor.unsqueeze(0).to(DEVICE)
    cam = cam_engine(img_dev, target_class)
    return cam.astype(np.float32)

class ScoreCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations = None
        self.target_layer = target_layer
        self.hook = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, _, __, output):
        self.activations = output.detach()

    def remove_hook(self):
        self.hook.remove()

    def __call__(self, img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        img = img_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _ = self.model(img)
        acts = self.activations[0]  
        C, h, w = acts.shape
        H, W = img.shape[-2:]

        acts_up = F.interpolate(acts.unsqueeze(0), size=(H, W),
                                mode="bilinear", align_corners=False)[0]  # [C,H,W]

        a_min = acts_up.view(C, -1).min(dim=1)[0].view(C, 1, 1)
        a_max = acts_up.view(C, -1).max(dim=1)[0].view(C, 1, 1)
        masks = (acts_up - a_min) / (a_max - a_min + 1e-8)  # [C,H,W]

        with torch.no_grad():
            base_score = F.softmax(self.model(img), dim=1)[0, target_class].item()

        weights = torch.zeros(C, device=DEVICE)
        with torch.no_grad():
            for i in range(0, C, SCORECAM_BATCH):
                m = masks[i:i+SCORECAM_BATCH]  
                masked = img * m.unsqueeze(1) 
                logits = self.model(masked)
                probs = F.softmax(logits, dim=1)[:, target_class]
                weights[i:i+SCORECAM_BATCH] = probs

        cam = (weights.view(C, 1, 1) * masks).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy().astype(np.float32)

def compute_scorecam(model: nn.Module, img_tensor: torch.Tensor,
                     target_class: int) -> np.ndarray:
    engine = ScoreCAM(model, model.resnet.layer4)
    try:
        cam = engine(img_tensor, target_class)
    finally:
        engine.remove_hook()
    return cam

def compute_lime(model: nn.Module, img_tensor: torch.Tensor,
                 target_class: int) -> np.ndarray:
    from lime import lime_image

    img_denorm = denorm(img_tensor)

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        return predict_proba_batch(model, images.astype(np.float32))

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    explanation = explainer.explain_instance(
        img_denorm.astype(np.float64),
        classifier_fn,
        top_labels=4,
        hide_color=0,
        num_samples=LIME_N_SAMPLES,
        segmentation_fn=lambda x: slic(
            x, n_segments=LIME_N_SEGMENTS, compactness=10,
            sigma=1, start_label=0
        ),
        random_seed=SEED,
    )

    segments = explanation.segments
    local_exp = dict(explanation.local_exp[target_class])
    attr_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in local_exp.items():
        attr_map[segments == seg_id] = weight

    return attr_map

def compute_shap(model: nn.Module, img_tensor: torch.Tensor,
                 target_class: int, background: torch.Tensor) -> np.ndarray:

    

    img_batch = img_tensor.unsqueeze(0).to(DEVICE)

    explainer = shap.GradientExplainer(model, background.to(DEVICE))

    # nsamples - liczba próbkowań w SHAP
    shap_values = explainer.shap_values(img_batch, nsamples=50)
    # Dla multi-class: shap_values to lista długości K, każda [1,3,H,W]
    # Wybieramy target_class i agregujemy po kanałach
    if isinstance(shap_values, list):
        sv = shap_values[target_class][0]  # [3,H,W]
    else:
        sv = shap_values[0, ..., target_class]  # nowe API SHAP

    # Suma wartości bezwzględnych po kanałach -> mapa [H,W]
    attr_map = np.abs(sv).sum(axis=0).astype(np.float32)
    return attr_map


# ============================================================================
# 5) Signed RISE (już w manuskrypcie - dla porównania)
# ============================================================================

def compute_signed_rise(model: nn.Module, img_tensor: torch.Tensor,
                        target_class: int,
                        n_masks: int = RISE_N_MASKS,
                        mask_res: int = RISE_MASK_RES,
                        keep_prob: float = RISE_KEEP_PROB) -> np.ndarray:
    """Signed RISE attribution map."""
    H, W = img_tensor.shape[-2:]
    img_batch = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        base_score = F.softmax(model(img_batch), dim=1)[0, target_class].item()

    attr = np.zeros((H, W), dtype=np.float32)
    cell_h = int(np.ceil(H / mask_res))
    cell_w = int(np.ceil(W / mask_res))
    up_h = (mask_res + 1) * cell_h
    up_w = (mask_res + 1) * cell_w

    rng = np.random.RandomState(SEED)
    batch_size = 32

    for i in range(0, n_masks, batch_size):
        b = min(batch_size, n_masks - i)
        low_res = (rng.rand(b, mask_res, mask_res) < keep_prob).astype(np.float32)
        masks = np.zeros((b, H, W), dtype=np.float32)
        for j in range(b):
            up = np.kron(low_res[j], np.ones((cell_h, cell_w), dtype=np.float32))
            sh = rng.randint(0, cell_h)
            sw = rng.randint(0, cell_w)
            masks[j] = up[sh:sh+H, sw:sw+W]

        masks_t = torch.from_numpy(masks).unsqueeze(1).to(DEVICE)  # [b,1,H,W]
        masked_imgs = img_batch * masks_t
        with torch.no_grad():
            scores = F.softmax(model(masked_imgs), dim=1)[:, target_class].cpu().numpy()

        signed = scores - base_score
        for j in range(b):
            attr += signed[j] * masks[j]

    attr /= (n_masks * keep_prob)
    return attr.astype(np.float32)

def insertion_curve(model: nn.Module, img_tensor: torch.Tensor,
                    attr_map: np.ndarray, target_class: int,
                    n_steps: int = INSERTION_DELETION_STEPS) -> np.ndarray:

    H, W = img_tensor.shape[-2:]
    img_dev = img_tensor.to(DEVICE)
    n_pixels = H * W

    flat_attr = attr_map.flatten()
    order = np.argsort(-flat_attr)

    step_size = n_pixels // n_steps
    scores = []

    blank = torch.zeros_like(img_dev)
    if USE_IMAGENET_NORM:
        mean = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=DEVICE).view(3, 1, 1)
        blank = (blank - mean) / std  # czarny po normalizacji

    current = blank.clone()

    model.eval()
    with torch.no_grad():
        for k in range(n_steps + 1):
            n_revealed = min(k * step_size, n_pixels)
            if n_revealed > 0:
                idx_to_reveal = order[:n_revealed]
                rows = idx_to_reveal // W
                cols = idx_to_reveal % W

                current = blank.clone()
                current[:, rows, cols] = img_dev[:, rows, cols]
            logits = model(current.unsqueeze(0))
            p = F.softmax(logits, dim=1)[0, target_class].item()
            scores.append(p)

    return np.array(scores, dtype=np.float32)


def deletion_curve(model: nn.Module, img_tensor: torch.Tensor,
                   attr_map: np.ndarray, target_class: int,
                   n_steps: int = INSERTION_DELETION_STEPS) -> np.ndarray:

    H, W = img_tensor.shape[-2:]
    img_dev = img_tensor.to(DEVICE)
    n_pixels = H * W

    flat_attr = attr_map.flatten()
    order = np.argsort(-flat_attr)

    step_size = n_pixels // n_steps
    scores = []
    blank_pixel_value = 0.0
    if USE_IMAGENET_NORM:
        blank_pixel = torch.tensor(
            (0.0 - IMAGENET_MEAN) / IMAGENET_STD, device=DEVICE, dtype=torch.float32
        ).view(3, 1)
    else:
        blank_pixel = torch.zeros(3, 1, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for k in range(n_steps + 1):
            n_removed = min(k * step_size, n_pixels)
            current = img_dev.clone()
            if n_removed > 0:
                idx_to_remove = order[:n_removed]
                rows = idx_to_remove // W
                cols = idx_to_remove % W
                current[:, rows, cols] = blank_pixel
            logits = model(current.unsqueeze(0))
            p = F.softmax(logits, dim=1)[0, target_class].item()
            scores.append(p)

    return np.array(scores, dtype=np.float32)


def auc_score(curve: np.ndarray) -> float:
    """AUC krzywej znormalizowanej do osi x w [0,1]."""
    n = len(curve)
    x = np.linspace(0, 1, n)
    return float(np.trapz(curve, x))


def faithfulness_correlation(model: nn.Module, img_tensor: torch.Tensor,
                             attr_map: np.ndarray, target_class: int,
                             n_patches: int = 50, patch_size: int = 30) -> float:

    H, W = img_tensor.shape[-2:]
    img_dev = img_tensor.to(DEVICE)
    blank_pixel = torch.tensor(
        (0.0 - IMAGENET_MEAN) / IMAGENET_STD if USE_IMAGENET_NORM else [0, 0, 0],
        device=DEVICE, dtype=torch.float32
    ).view(3, 1, 1)

    model.eval()
    with torch.no_grad():
        base_score = F.softmax(model(img_dev.unsqueeze(0)), dim=1)[0, target_class].item()

    rng = np.random.RandomState(SEED)
    attrs_sum = []
    score_drops = []

    with torch.no_grad():
        for _ in range(n_patches):
            r = rng.randint(0, H - patch_size)
            c = rng.randint(0, W - patch_size)
            # Suma atrybucji w patchu
            patch_attr = attr_map[r:r+patch_size, c:c+patch_size].sum()
            attrs_sum.append(patch_attr)

            current = img_dev.clone()
            current[:, r:r+patch_size, c:c+patch_size] = blank_pixel
            score = F.softmax(model(current.unsqueeze(0)), dim=1)[0, target_class].item()
            score_drops.append(base_score - score)

    attrs_sum = np.array(attrs_sum)
    score_drops = np.array(score_drops)

    if np.std(attrs_sum) < 1e-8 or np.std(score_drops) < 1e-8:
        return 0.0
    corr, _ = pearsonr(attrs_sum, score_drops)
    return float(corr)

METHODS = {
    "Grad-CAM": compute_gradcam,
    "Score-CAM": compute_scorecam,
    "LIME": compute_lime,
    "SHAP": "shap_special",   # potrzebuje background
    "RISE": compute_signed_rise,
    "Signed-RISE": compute_signed_rise,  # ten sam algorytm, ale traktowany jako baseline
}

def run_benchmark(model: nn.Module, samples: List[Tuple[torch.Tensor, int]],
                  background: torch.Tensor) -> pd.DataFrame:
    results = []

    for sample_idx, (img_tensor, true_label) in enumerate(samples):
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
            pred_class = int(logits.argmax(dim=1).item())

        target = pred_class

        print(f"\n[{sample_idx+1}/{len(samples)}] true={CLASS_NAMES[true_label]} "
              f"pred={CLASS_NAMES[pred_class]}")

        for method_name in ["Grad-CAM", "Score-CAM", "LIME", "SHAP", "Signed-RISE"]:
            try:
                if method_name == "SHAP":
                    attr = compute_shap(model, img_tensor, target, background)
                elif method_name == "LIME":
                    attr = compute_lime(model, img_tensor, target)
                elif method_name == "Score-CAM":
                    attr = compute_scorecam(model, img_tensor, target)
                elif method_name == "Grad-CAM":
                    attr = compute_gradcam(model, img_tensor, target)
                elif method_name == "Signed-RISE":
                    attr = compute_signed_rise(model, img_tensor, target)
                else:
                    continue

                attr_norm = normalize_attr(attr)

                ins_curve = insertion_curve(model, img_tensor, attr_norm, target)
                del_curve = deletion_curve(model, img_tensor, attr_norm, target)
                ins_auc = auc_score(ins_curve)
                del_auc = auc_score(del_curve)
                faith = faithfulness_correlation(model, img_tensor, attr_norm, target)

                results.append({
                    "sample_idx": sample_idx,
                    "true_class": CLASS_NAMES[true_label],
                    "pred_class": CLASS_NAMES[pred_class],
                    "method": method_name,
                    "insertion": ins_auc,
                    "deletion": del_auc,
                    "faithfulness": faith,
                })
                print(f"  {method_name:12s}  ins={ins_auc:.3f}  del={del_auc:.3f}  faith={faith:+.3f}")

            except Exception as e:
                print(f"  {method_name:12s}  BŁĄD: {e}")
                results.append({
                    "sample_idx": sample_idx,
                    "true_class": CLASS_NAMES[true_label],
                    "pred_class": CLASS_NAMES[pred_class],
                    "method": method_name,
                    "insertion": np.nan,
                    "deletion": np.nan,
                    "faithfulness": np.nan,
                })

    return pd.DataFrame(results)

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("method").agg(
        insertion_mean=("insertion", "mean"),
        insertion_std=("insertion", "std"),
        deletion_mean=("deletion", "mean"),
        deletion_std=("deletion", "std"),
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_std=("faithfulness", "std"),
        n=("insertion", "count"),
    ).round(3)
    return summary


def pairwise_wilcoxon(df: pd.DataFrame, metric: str = "insertion",
                      reference: str = "Signed-RISE") -> pd.DataFrame:
    ref_vals = df[df.method == reference].sort_values("sample_idx")[metric].values
    rows = []
    for m in df.method.unique():
        if m == reference:
            continue
        vals = df[df.method == m].sort_values("sample_idx")[metric].values
        mask = ~(np.isnan(ref_vals) | np.isnan(vals))
        if mask.sum() < 5:
            rows.append({"method": m, "p_value": np.nan, "n": int(mask.sum())})
            continue
        try:
            stat, p = wilcoxon(ref_vals[mask], vals[mask])
        except ValueError as e:
            stat, p = np.nan, np.nan
        rows.append({"method": m, "p_value": p, "n": int(mask.sum())})
    return pd.DataFrame(rows)

def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")

    import glob, re
    ckpt_paths = sorted(glob.glob(os.path.join("checkpoints", "captrad-*.ckpt")))
    best_path, best_f1 = None, -1.0
    for p in ckpt_paths:
        m = re.search(r"val_f1=([\d]+\.[\d]+)", p)
        if m and float(m.group(1)) > best_f1:
            best_f1 = float(m.group(1))
            best_path = p
    if best_path is None:
        print("[BŁĄD] Brak checkpointu! Uruchom najpierw główny skrypt treningowy.")
        sys.exit(1)
    print(f"[INFO] Ładuję checkpoint: {best_path}")
    model = HybridCNN.load_from_checkpoint(best_path).to(DEVICE).eval()

    samples = build_stratified_subset(DATASET_PATH, n_per_class=N_SAMPLES_PER_CLASS)

    print(f"[INFO] Buduję background SHAP ({SHAP_N_BACKGROUND} obrazów)")
    bg_indices = np.random.RandomState(SEED + 1).choice(
        len(samples), size=min(SHAP_N_BACKGROUND, len(samples)), replace=False
    )
    background = torch.stack([samples[i][0] for i in bg_indices])

    print(f"\n[INFO] Start benchmark: {len(samples)} obrazów × 5 metod\n")
    df = run_benchmark(model, samples, background)

    raw_path = os.path.join(OUTPUT_DIR, "xai_raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\n[OK] Surowe wyniki: {raw_path}")

    summary = summarize_results(df)
    summary_path = os.path.join(OUTPUT_DIR, "xai_summary.csv")
    summary.to_csv(summary_path)
    print("\n=== Podsumowanie (mean ± std) ===")
    print(summary.to_string())

    print("\n=== Wilcoxon signed-rank vs Signed-RISE ===")
    for metric in ["insertion", "deletion", "faithfulness"]:
        wilc = pairwise_wilcoxon(df, metric=metric, reference="Signed-RISE")
        print(f"\n[{metric}]")
        print(wilc.to_string(index=False))
        wilc.to_csv(os.path.join(OUTPUT_DIR, f"wilcoxon_{metric}.csv"), index=False)

    print("\n=== Gotowy fragment do manuskryptu (LaTeX-friendly) ===")
    for method in summary.index:
        ins = summary.loc[method, "insertion_mean"]
        delv = summary.loc[method, "deletion_mean"]
        fai = summary.loc[method, "faithfulness_mean"]
        print(f"  {method}: Insertion {ins:.3f}, Deletion {delv:.3f}, Faithfulness {fai:.3f}")

if __name__ == "__main__":
    main()