# -*- coding: utf-8 -*-
# CapsNet na ADNI + metryki + Grad-CAM (styl ENGRAP) — GPU, szybki I/O, Spyder-safe

import os, cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")  # zapis do plików bez otwierania okien
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import itertools

# ========================== KONFIG ==========================
DATA_DIR   = r"F:\Badania\embedded\Dataset"   # ImageFolder: podkatalogi = klasy
IMG_SIZE   = 299                               # szybkie i bezpieczne dla CapsNet
BATCH_SIZE = 32
MAX_EPOCHS = 50
N_CLASSES  = 4
USE_IMAGENET_NORM = False                     # MRI: zwykle False
OUT_METRICS_CSV = "val_metrics_capsnet.csv"
OUT_GRID_PNG    = "capsnet_gradcam_grid.png"

# Przyspieszenie macierzy na GPU
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ==================== Transformy i denormalizacja ====================
if USE_IMAGENET_NORM:
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    def denorm(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(MEAN, dtype=x.dtype, device=x.device).reshape(3,1,1)
        std  = torch.tensor(STD,  dtype=x.dtype, device=x.device).reshape(3,1,1)
        return (x*std + mean).clamp(0,1)
else:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # [0,1]
    ])
    def denorm(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0,1)

# ==================== Capsule Layer z routowaniem ====================
class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, routing_iters=1):
        super().__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        # [1, Cin, Cout, Dout, Din]
        self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    def forward(self, x):  # x: [B, Cin, Din]
        B = x.size(0)
        x_exp = x.unsqueeze(2).unsqueeze(4)             # [B,Cin,1,Din,1]
        W = self.W.expand(B, *self.W.shape[1:])         # [B,Cin,Cout,Dout,Din]
        u_hat = torch.matmul(W, x_exp).squeeze(-1)      # [B,Cin,Cout,Dout]

        b = torch.zeros(B, self.in_capsules, self.out_capsules, 1, device=x.device)
        for i in range(self.routing_iters):
            c = F.softmax(b, dim=2)                     # [B,Cin,Cout,1]
            s = (c * u_hat).sum(dim=1, keepdim=True)    # [B,1,Cout,Dout]
            v = self.squash(s)                          # [B,1,Cout,Dout]
            if i < self.routing_iters - 1:
                b = b + (u_hat * v).sum(-1, keepdim=True)
        return v.squeeze(1)                              # [B,Cout,Dout]

    @staticmethod
    def squash(s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1.0 + norm**2)) * (s / (norm + 1e-8))

# ==================== PureCapsNet (Lightning) — szybka wersja ====================
class PureCapsNet(pl.LightningModule):
    def __init__(self, n_classes=N_CLASSES, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()

        # Downsampling: 96 -> 48 -> 24 -> primary 12x12
        self.conv1 = nn.Conv2d(3, 256, kernel_size=9, stride=2, padding=4)
        self.pool1 = nn.MaxPool2d(2)
        # 16 kapsuł po 8D = 128 kanałów
        self.primary_caps = nn.Conv2d(256, 8 * 16, kernel_size=9, stride=2, padding=4)

        self.caps_layer = None
        self.fc = None
        self.val_outputs = []

    def forward(self, x):
        # Bez channels_last — unikamy błędów rangi; używamy reshape/contiguous
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.primary_caps(x)           # [B, 128, H, W]  (C = 16*8)

        B, C, H, W = x.shape
        cap_dim = 8
        assert C % cap_dim == 0
        num_caps = C // cap_dim

        # [B,C,H,W] -> [B,num_caps,cap_dim,H,W] -> [B,num_caps*H*W,cap_dim]
        x = x.reshape(B, num_caps, cap_dim, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(B, num_caps * H * W, cap_dim)

        if self.caps_layer is None:
            self.caps_layer = CapsuleLayer(
                in_capsules=num_caps * H * W, in_dim=cap_dim,
                out_capsules=self.hparams.n_classes, out_dim=8,
                routing_iters=1
            ).to(x.device)
            self.fc = nn.Linear(self.hparams.n_classes * 8, self.hparams.n_classes).to(x.device)

        x = self.caps_layer(x)             # [B, n_classes, 8]
        x = x.reshape(B, -1)
        logits = self.fc(x)                # [B, n_classes]
        return logits

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)
        loss   = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_outputs.append((y.cpu(), preds.cpu(), probs.cpu()))

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        ys, preds, probs = zip(*self.val_outputs)
        y = torch.cat(ys).numpy()
        y_hat = torch.cat(preds).numpy()
        p = torch.cat(probs).numpy()
        self.val_outputs.clear()

        acc = accuracy_score(y, y_hat)
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="macro", zero_division=0)
        try:
            y_oh = F.one_hot(torch.tensor(y), num_classes=p.shape[1]).numpy()
            auc_macro = roc_auc_score(y_oh, p, average="macro", multi_class="ovr")
        except Exception:
            auc_macro = float("nan")

        df = pd.DataFrame([{"Accuracy":acc, "Precision":prec, "Recall":rec, "F1":f1, "AUC":auc_macro}])
        df.to_csv(OUT_METRICS_CSV, index=False)
        print("\n=== WALIDACJA (macro) ===")
        print(df.to_string(index=False))
        print(f"\nLaTeX: CapsNet & {acc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {auc_macro:.3f} \\\n")
        
        # --- Confusion Matrix jako obraz do TensorBoard ---
        cm = confusion_matrix(y, y_hat, labels=list(range(p.shape[1])))
        fig_cm = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig_cm.add_subplot(111)
        ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion matrix (val)')
        tick_marks = np.arange(p.shape[1])
        ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
        ax.set_xlabel('Pred'); ax.set_ylabel('True')
        # liczby na kratkach
        thresh = cm.max() / 2.0 if cm.size else 0.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, int(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
        fig_cm.tight_layout()
        tb = self.logger.experiment  # SummaryWriter
        tb.add_figure("val/confusion_matrix", fig_cm, global_step=self.global_step)
        plt.close(fig_cm)
        
        # opcjonalnie zapisz też metryki jako scalars (masz już CSV/print)
        tb.add_scalar("val/Accuracy", acc, self.global_step)
        tb.add_scalar("val/Precision_macro", prec, self.global_step)
        tb.add_scalar("val/Recall_macro", rec, self.global_step)
        tb.add_scalar("val/F1_macro", f1, self.global_step)
        if not np.isnan(auc_macro):
            tb.add_scalar("val/AUC_macro", auc_macro, self.global_step)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        return opt

# ==================== Grad-CAM (styl ENGRAP) — autocast OFF ====================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients  = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, out):   self.activations = out          # [B,C,Hf,Wf]
    def _bwd_hook(self, _, grad_in, grad_out): self.gradients = grad_out[0]  # [B,C,Hf,Wf]

    def __call__(self, x: torch.Tensor, target_idx: int = None) -> np.ndarray:
        # Wyjaśnianie bez AMP i w FP32, aby uniknąć konfliktów typów
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.model(x.float())                 # [1,C]
            if target_idx is None:
                target_idx = int(logits.argmax(dim=1).item())
            score = logits[0, target_idx]
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)

        A  = self.activations[0]                           # [C,Hf,Wf]
        dA = self.gradients[0]                             # [C,Hf,Wf]
        w  = dA.mean(dim=(1,2), keepdim=True)              # [C,1,1]
        cam = torch.relu((w * A).sum(dim=0))               # [Hf,Wf]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        H, W = x.shape[-2:]
        cam = F.interpolate(cam[None,None], size=(H,W), mode="bilinear", align_corners=False)[0,0]
        return cam.detach().cpu().numpy()

def create_brain_mask(img01: torch.Tensor) -> np.ndarray:
    img = (denorm(img01).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(gray[thr==255]) < np.mean(gray[thr==0]):
        thr = cv2.bitwise_not(thr)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thr,  cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask,  cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (lab == largest).astype(np.uint8)*255
    return mask

def create_brain_mask(img01: torch.Tensor) -> np.ndarray:
    img  = (denorm(img01).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # łagodny próg + wypełnianie dziur
    _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    # fill holes (flood fill od brzegu)
    h, w = mask.shape
    ff = np.zeros((h+2, w+2), np.uint8)
    inv = 255 - mask
    cv2.floodFill(inv, ff, (0,0), 0)
    mask = 255 - inv
    return mask

def overlay_masked(img01: torch.Tensor, cam01: np.ndarray, alpha=0.45) -> np.ndarray:
    import matplotlib
    cmap   = matplotlib.colormaps.get("jet")
    hm_rgb = cmap(cam01)[..., :3]
    img    = denorm(img01).permute(1,2,0).cpu().numpy()
    blend  = (1 - alpha) * img + alpha * hm_rgb
    brain  = create_brain_mask(img01).astype(np.float32) / 255.0
    # zamiast wycinać – tylko przyciemnij tło do 40%
    blend  = blend * (0.4 + 0.6 * brain[..., None])
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)


def make_gradcam_grid_capsnet(model: nn.Module, loader: DataLoader, save_path=OUT_GRID_PNG, n_images=8):
    device = next(model.parameters()).device
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n_images].to(device, non_blocking=True).float().contiguous()
    labels = labels[:n_images].to(device, non_blocking=True)

    assert hasattr(model, "primary_caps"), "Model CapsNet musi mieć atrybut 'primary_caps'."
    gc = GradCAM(model, model.primary_caps)

    with torch.enable_grad():
        # wyjaśnianie bez AMP i w FP32
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(images.float())
        targets = logits.argmax(dim=1)  # używamy predykcji

    cams = [gc(images[i:i+1], int(targets[i].item())) for i in range(images.size(0))]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=200)
    axes = axes.ravel()
    for i in range(len(cams)):
        vis = overlay_masked(images[i].detach().cpu(), cams[i])
        axes[i].imshow(vis); axes[i].axis("off")
        axes[i].set_title(f"Pred: {int(targets[i])} | True: {int(labels[i])}", fontsize=8)
    for j in range(len(cams), 8):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[OK] zapisano: {save_path}")

def main():
    print("cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("torch.version.cuda:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.is_available())

    # --- DATASET + LOADERY (Windows -> muszą być w main przy num_workers>0) ---
    assert os.path.isdir(DATA_DIR), f"Nie znaleziono katalogu: {DATA_DIR}"
    full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    n_train = int(0.8 * len(full_ds))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    # --- MODEL NA GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PureCapsNet(n_classes=N_CLASSES).to(device)

    logger = TensorBoardLogger(save_dir="tb_logs", name="capsnet_gradcam")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    # --- PRZYGOTOWANIE DO GRAD-CAM (model na GPU, FP32, z gradientami) ---
    model.eval()
    model.to(device)
    model.float()
    for p in model.parameters():
        p.requires_grad_(True)

    # --- HEATMAPY 2x4 ---
    make_gradcam_grid_capsnet(model, val_loader, save_path=OUT_GRID_PNG, n_images=8)
    print("Done.")
    
    trainer.validate(model, val_loader)

    # DODAJ TU:
    torch.save(model, 'capsnet_model.pth')
    print(f"Model saved to: capsnet_model.pth")

    model.eval().to(device).float()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # wymagane na Windows przy num_workers>0
    main()
