import os, cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import itertools

DATA_DIR   = r"F:\Badania\embedded\Dataset"   


IMG_SIZE   = 299
BATCH_SIZE = 32
MAX_EPOCHS = 50
N_CLASSES  = 4
USE_IMAGENET_NORM = False    # dla spójności z CapsNet
OUT_METRICS_CSV = "val_metrics_resnet50.csv"
OUT_GRID_PNG    = "resnet50_gradcam_grid.png"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

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

class PureResNet50(pl.LightningModule):
    def __init__(self, n_classes=N_CLASSES, lr=3e-4, use_imagenet_init=False):
        super().__init__()
        self.save_hyperparameters()
        # bez pretrainu żeby zostać przy tym samym preprocessie co CapsNet
        self.backbone = models.resnet50(weights=None if not use_imagenet_init else models.ResNet50_Weights.IMAGENET1K_V1)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, n_classes)

        self.val_outputs = []

    def forward(self, x):
        return self.backbone(x)

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
        print(f"\nLaTeX: ResNet50 & {acc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} & {auc_macro:.3f} \\\n")
        
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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)

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
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.model(x.float())
            if target_idx is None:
                target_idx = int(logits.argmax(dim=1).item())
            score = logits[0, target_idx]
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)

        A  = self.activations[0]                     # [C,Hf,Wf]
        dA = self.gradients[0]                       # [C,Hf,Wf]
        w  = dA.mean(dim=(1,2), keepdim=True)        # [C,1,1]
        cam = torch.relu((w * A).sum(dim=0))         # [Hf,Wf]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        H, W = x.shape[-2:]
        cam = F.interpolate(cam[None,None], size=(H,W), mode="bilinear", align_corners=False)[0,0]
        cam = cam.detach().cpu().numpy()
        
        # --- post-processing: łagodne wygładzenie + podbicie kontrastu ---
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=1.0, sigmaY=1.0)
        gamma = 0.8                         # <1.0: jaśniejsze obszary istotne
        cam = np.clip(cam, 0.0, 1.0) ** gamma
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def overlay_heatmap(img01: torch.Tensor, cam01: np.ndarray, alpha=0.45) -> np.ndarray:
    import matplotlib
    cmap   = matplotlib.colormaps.get("jet")
    hm_rgb = cmap(cam01)[..., :3]
    img    = denorm(img01).permute(1,2,0).cpu().numpy()
    blend  = (1 - alpha)*img + alpha*hm_rgb
    return (np.clip(blend, 0, 1) * 255).astype(np.uint8)

def make_gradcam_grid_resnet(model: nn.Module, loader: DataLoader, save_path=OUT_GRID_PNG, n_images=8):
    device = next(model.parameters()).device
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n_images].to(device, non_blocking=True).float().contiguous()
    labels = labels[:n_images].to(device, non_blocking=True)

    # target = ostatnia konwolucja w layer4
    target_layer = model.backbone.layer3[-1].conv3 if hasattr(model, "backbone") else model.layer4[-1].conv3
    
    gc = GradCAM(model, target_layer)

    with torch.enable_grad():
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(images.float())
        targets = logits.argmax(dim=1)

    cams = [gc(images[i:i+1], int(targets[i].item())) for i in range(images.size(0))]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=200)
    axes = axes.ravel()
    
    for i in range(len(cams)):
        vis = overlay_heatmap(images[i].detach().cpu(), cams[i])   # albo overlay_masked(...)
        ax = axes[i]
        ax.imshow(vis)
        ax.axis("off")
        # mniejsza czcionka + odstęp od górnej krawędzi
        ax.set_title(f"Pred: {int(targets[i])} | True: {int(labels[i])}",
                     fontsize=10, pad=6)
    
    for j in range(len(cams), 8):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
            
    print(f"[OK] zapisano: {save_path}")

def main():
    print("cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("torch.version.cuda:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.is_available())

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PureResNet50(n_classes=N_CLASSES, lr=3e-4).to(device)
    
    logger = TensorBoardLogger(save_dir="tb_logs", name="resnet50_gradcam")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)

    model.eval().to(device).float()
    for p in model.parameters():
        p.requires_grad_(True)

    make_gradcam_grid_resnet(model, val_loader, save_path=OUT_GRID_PNG, n_images=8)
    print("Done.")

    trainer.validate(model, val_loader)

    torch.save(model, 'resnet50_model.pth')
    print("Model saved to: resnet50_model.pth")

    model.eval().to(device).float()
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()   # Windows + num_workers>0
    main()

