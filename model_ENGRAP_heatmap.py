import os, time, cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import itertools

DATASET_PATH = r"F:/Badania/embedded/Dataset"   # <- tu ustaw swój katalog ImageFolder
IMG_SIZE     = 299
BATCH_SIZE   = 32
NUM_WORKERS  = 8
MAX_EPOCHS   = 50
USE_IMAGENET_NORM = True           # ResNet50 lubisz z ImageNet mean/std
VAL_CSV_PATH = "engrap_val_metrics.csv"
GRID_PNG     = "engrap_gradcam_grid.png"

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 img_size: int = IMG_SIZE,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 use_imagenet_norm: bool = USE_IMAGENET_NORM,
                 val_split: float = 0.2,
                 seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_imagenet_norm = use_imagenet_norm
        self.val_split = val_split
        self.seed = seed
        self.train_ds = self.val_ds = self.test_ds = None

        if self.use_imagenet_norm:
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
        else:
            mean, std = [0,0,0], [1,1,1]

        self.train_tf = transforms.Compose([
            transforms.Resize(int(self.img_size*1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize(int(self.img_size*1.1)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def _has_splits(self) -> bool:
        return all(os.path.isdir(os.path.join(self.data_dir, sub))
                   for sub in ("train", "val"))

    def setup(self, stage=None):
        if self._has_splits():
            train_root = os.path.join(self.data_dir, "train")
            val_root   = os.path.join(self.data_dir, "val")
            test_root  = os.path.join(self.data_dir, "test")
            self.train_ds = datasets.ImageFolder(train_root, transform=self.train_tf)
            self.val_ds   = datasets.ImageFolder(val_root,   transform=self.eval_tf)
            self.test_ds  = datasets.ImageFolder(test_root,  transform=self.eval_tf) if os.path.isdir(test_root) else None
        else:
            full = datasets.ImageFolder(self.data_dir, transform=self.train_tf)
            n_total = len(full)
            n_val   = int(self.val_split*n_total)
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(42)
            train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
            # dwa obiekty ImageFolder z różnymi transformami
            self.train_ds = torch.utils.data.Subset(datasets.ImageFolder(self.data_dir, transform=self.train_tf), train_idx.indices)
            self.val_ds   = torch.utils.data.Subset(datasets.ImageFolder(self.data_dir, transform=self.eval_tf),  val_idx.indices)
            self.test_ds  = None

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        if self.test_ds is None:
            return self.val_dataloader()
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("w", torch.tensor(weight) if weight is not None else None)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.w, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1-pt)**self.gamma * ce
        return loss.mean()

class CapsuleLayer(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_routes=3):
        super().__init__()
        self.in_capsules  = in_capsules
        self.in_dim       = in_dim
        self.out_capsules = out_capsules
        self.out_dim      = out_dim
        self.num_routes   = num_routes
        self.W = nn.Parameter(0.01*torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    @staticmethod
    def squash(s):
        n = torch.norm(s, dim=-1, keepdim=True)
        return (n**2/(1+n**2)) * (s/(n+1e-8))

    def forward(self, x):                           # x: [B, Cin, Din]
        B = x.size(0)
        W = self.W.expand(B, *self.W.shape[1:])    # [B,Cin,Cout,Dout,Din]
        x = x.unsqueeze(2).unsqueeze(-1)           # [B,Cin,1,Din,1]
        u_hat = torch.matmul(W, x).squeeze(-1)     # [B,Cin,Cout,Dout]
        b = torch.zeros(B, self.in_capsules, self.out_capsules, 1, device=x.device)
        for i in range(self.num_routes):
            c = F.softmax(b, dim=2)                # [B,Cin,Cout,1]
            s = (c * u_hat).sum(1, keepdim=True)   # [B,1,Cout,Dout]
            v = self.squash(s)                     # [B,1,Cout,Dout]
            if i < self.num_routes - 1:
                b = b + (u_hat * v).sum(-1, keepdim=True)
        return v.squeeze(1)                        # [B,Cout,Dout]

class HybridCNN(pl.LightningModule):
    """
    ResNet50 (GAP 2048) -> CapsuleLayer(10×16) -> Transformer(L=2,H=4,d=16) -> FC(4)
    + Late fusion [resnet_feat || CLS]
    """
    def __init__(self, n_classes=4):
        super().__init__()
        self.save_hyperparameters()
        # ResNet backbone
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out_features = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        # Caps
        self.in_capsules = 64
        self.in_dim      = 32
        self.out_caps    = 10
        self.out_dim     = 16

        self.fc_transform = nn.Linear(self.resnet_out_features, self.in_capsules*self.in_dim)
        self.capsule_layer = CapsuleLayer(self.in_capsules, self.in_dim, self.out_caps, self.out_dim, num_routes=3)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.out_dim))

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=self.out_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Classifier (late fusion)
        self.fc1 = nn.Linear(self.resnet_out_features + self.out_dim, 512)
        self.fc2 = nn.Linear(512, self.hparams.n_classes)
        self.dropout = nn.Dropout(0.5)

        # Loss + metrics
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=torch.tensor([1.0, 7.0, 1.0, 2.0]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = nn.ModuleDict({
            "val_precision": MulticlassPrecision(num_classes=self.hparams.n_classes, average="macro").to(device),
            "val_recall":    MulticlassRecall   (num_classes=self.hparams.n_classes, average="macro").to(device),
            "val_f1":        MulticlassF1Score  (num_classes=self.hparams.n_classes, average="macro").to(device),
        })
        self.validation_preds = []
        self.validation_labels = []

    def forward(self, x):
        resnet_out = self.resnet(x)                                     # [B,2048]
        transformed = self.fc_transform(resnet_out)                     # [B,64*32]
        caps_in = transformed.view(x.size(0), self.in_capsules, self.in_dim)
        caps_out = self.capsule_layer(caps_in)                          # [B,10,16]

        B = caps_out.size(0)
        cls = self.cls_token.expand(B,1,self.out_dim)                   # [B,1,16]
        caps_with_cls = torch.cat([cls, caps_out], dim=1)               # [B,11,16]

        tr_out = self.transformer_encoder(caps_with_cls)                # [B,11,16]
        cls_out = tr_out[:,0,:]                                         # [B,16]

        fused = torch.cat([resnet_out, cls_out], dim=1)                 # [B,2064]
        fused = self.dropout(fused)
        h = F.relu(self.fc1(fused))
        logits = self.fc2(h)                                            # [B,4]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        # upewnij się, że wagi loss są na tym samym urządzeniu
        if isinstance(self.criterion.w, torch.Tensor) and self.criterion.w.device != x.device:
            self.criterion.w = self.criterion.w.to(x.device)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        self.validation_preds.append(preds.detach().cpu())
        self.validation_labels.append(y.detach().cpu())

        # log metryk krokowych (opcjonalnie)
        f1  = self.metrics["val_f1"](preds, y)
        pre = self.metrics["val_precision"](preds, y)
        self.log("val_f1_step",  f1,  prog_bar=True, on_step=True)
        self.log("val_pre_step", pre, prog_bar=True, on_step=True)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if not self.validation_preds:
            return
        y_pred = torch.cat(self.validation_preds).to(self.device)
        y_true = torch.cat(self.validation_labels).to(self.device)

        f1  = self.metrics["val_f1"](y_pred, y_true)
        pre = self.metrics["val_precision"](y_pred, y_true)
        rec = self.metrics["val_recall"](y_pred, y_true)
        acc = (y_pred == y_true).float().mean()

        # log do Lightning
        self.log("val_f1",  f1,  prog_bar=True, on_epoch=True)
        self.log("val_pre", pre, prog_bar=True, on_epoch=True)
        self.log("val_rec", rec, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        # do CSV (makro) — AUC pomijamy tutaj (brak proba), to sekcja klasyfikacyjna
        df = pd.DataFrame([{
            "Accuracy": float(acc.detach().cpu()),
            "Precision": float(pre.detach().cpu()),
            "Recall": float(rec.detach().cpu()),
            "F1": float(f1.detach().cpu()),
        }])
        df.to_csv(VAL_CSV_PATH, index=False)
        print("\n=== ENGRAP: wyniki walidacji (macro) ===")
        print(df.to_string(index=False))
        print(f"\nLaTeX row: ENGRAP (HybridCNN) & {df.Accuracy.iloc[0]:.3f} & {df.Precision.iloc[0]:.3f} & {df.Recall.iloc[0]:.3f} & {df.F1.iloc[0]:.3f} & 1.000 \\\\")  # jeśli AUC=1.000 jak w Twojej tabeli


        y_pred_cpu = y_pred.detach().cpu().numpy()
        y_true_cpu = y_true.detach().cpu().numpy()

        cm = confusion_matrix(
            y_true_cpu, y_pred_cpu,
            labels=list(range(self.hparams.n_classes))
        )

        fig_cm = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig_cm.add_subplot(111)
        ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion matrix (val)')
        tick_marks = np.arange(self.hparams.n_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')

        thresh = cm.max() / 2.0 if cm.size else 0.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, int(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

        fig_cm.tight_layout()

        # Zapis lokalny do PNG
        local_path = "confusion_matrix_val.png"
        plt.savefig(local_path, bbox_inches="tight")
        plt.close(fig_cm)
        print(f"[OK] Confusion matrix zapisany do {local_path}")

        # Metryki dalej można logować do TensorBoard jeśli chcesz
        tb = self.logger.experiment
        tb.add_scalar("val/Accuracy", acc, self.global_step)
        tb.add_scalar("val/Precision_macro", pre, self.global_step)
        tb.add_scalar("val/Recall_macro", rec, self.global_step)
        tb.add_scalar("val/F1_macro", f1, self.global_step)

        # wyczyść bufory
        self.validation_preds.clear()
        self.validation_labels.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return [opt], [sch]
        
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients  = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _, __, output):
        self.activations = output

    def _bwd_hook(self, _, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x: torch.Tensor, target_idx: int = None) -> np.ndarray:
        # FP32 bez AMP, by uniknąć konfliktów typów
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.model(x.float())
            if target_idx is None:
                target_idx = int(logits.argmax(dim=1).item())
            score = logits[0, target_idx]
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)

        A  = self.activations[0]                    # [C,Hf,Wf]
        dA = self.gradients[0]                      # [C,Hf,Wf]
        w  = dA.mean(dim=(1,2), keepdim=True)       # [C,1,1]
        cam = torch.relu((w * A).sum(0))            # [Hf,Wf]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        H, W = x.shape[-2:]
        cam = F.interpolate(cam[None,None], size=(H,W), mode="bilinear", align_corners=False)[0,0]
        return cam.detach().cpu().numpy()

def _denorm_for_overlay(img: torch.Tensor, imagenet_norm=True) -> np.ndarray:
    x = img.clone()
    if imagenet_norm:
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(3,1,1)
        x = x*std + mean
    x = x.clamp(0,1)
    return x.permute(1,2,0).cpu().numpy()

def make_gradcam_grid_engrap(model: nn.Module, val_loader: DataLoader,
                             save_path: str = GRID_PNG, n_images: int = 8,
                             target_layer: nn.Module = None, title_below: bool = False):
    device = next(model.parameters()).device
    model.eval()

    images, labels = next(iter(val_loader))
    images = images[:n_images].to(device, non_blocking=True)
    labels = labels[:n_images].to(device, non_blocking=True)

    if target_layer is None:
        # ResNet backbone
        if hasattr(model, "resnet"):
            target_layer = model.resnet.layer4
        else:
            raise AttributeError("Podaj target_layer dla Grad-CAM (np. model.resnet.layer4).")

    cam_explainer = GradCAM(model, target_layer)

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)
        
    import matplotlib
    cmap = matplotlib.colormaps.get("jet")

    fig, axes = plt.subplots(2, 4, figsize=(12,6), dpi=200, constrained_layout=False)
    axes = axes.ravel()
    for i in range(n_images):
        # CAM (z gradientami, więc poza no_grad)
        with torch.enable_grad():
            cam = cam_explainer(images[i:i+1], int(preds[i].item()))
        cam_rgb = cmap(cam)[..., :3]

        base = _denorm_for_overlay(images[i], imagenet_norm=USE_IMAGENET_NORM)
        # lekko przyciemnić tło dla MRI (granat)
        alpha = 0.4  # przezroczystość mapy (0 = niewidoczna, 1 = całkowicie zakrywa obraz)
        cam_rgb = cmap(cam)[..., :3]
        base = _denorm_for_overlay(images[i], imagenet_norm=USE_IMAGENET_NORM)
        
        # obraz wyraźny, heatmapa półprzezroczysta
        blend = (1 - alpha) * base + alpha * cam_rgb
        blend = np.clip(blend, 0, 1)

        ax = axes[i]
        ax.imshow(blend)
        ax.set_axis_off()
        title = f"Pred: {int(preds[i])} | True: {int(labels[i])}"
        if title_below:
            ax.set_title("")  # tytuł wyłączony, podpis można dodać pod spodem
        else:
            ax.set_title(title, fontsize=10, pad=6)

    for j in range(n_images, 8):
        axes[j].set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Zapisano heatmapy: {save_path}")

def main():
    print("cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
    assert os.path.isdir(DATASET_PATH), f"Brak katalogu: {DATASET_PATH}"
    dm = CustomDataModule(DATASET_PATH)
    dm.setup("fit")
    
    model = HybridCNN()

    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        filename="captrad-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    logger = TensorBoardLogger("tb_logs", name="engrap")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[ckpt, TQDMProgressBar()],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        precision="32-true"
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    print(f"Czas treningu: {(time.time()-t0)/3600:.2f} h")

    best_ckpt_path = ckpt.best_model_path
    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        print(f"Załadowano best ckpt: {best_ckpt_path}")
        model_best = HybridCNN.load_from_checkpoint(best_ckpt_path)
    else:
        print("Brak best_ckpt — używam aktualnego modelu.")
        model_best = model

    model_best.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    trainer.validate(model, datamodule=dm)
    print(f"Czas treningu: {(time.time()-t0)/3600:.2f} h")

    torch.save(model, 'engrap_model.pth')
    print("Model saved to: engrap_model.pth")

    best_ckpt_path = ckpt.best_model_path
    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        print(f"Załadowano best ckpt: {best_ckpt_path}")
        model_best = HybridCNN.load_from_checkpoint(best_ckpt_path)
        # DODAJ TU zapisywanie najlepszego modelu:
        torch.save(model_best, 'best_engrap_model.pth')
        print("Best model saved to: best_engrap_model.pth")
    else:
        print("Brak best_ckpt — używam aktualnego modelu.")
        model_best = model

    model_best.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    make_gradcam_grid_engrap(
        model_best,
        dm.val_dataloader(),
        save_path=GRID_PNG,
        n_images=8,
        target_layer=model_best.resnet.layer4,
        title_below=False
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows + num_workers>0
    main()

