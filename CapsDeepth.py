# ====== ENGRAP: warianty głębokości kapsuł (1/2/3 warstwy) + zapis metryk ======
import os, time, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
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

    def forward(self, x):                           
        B = x.size(0)
        W = self.W.expand(B, *self.W.shape[1:])    
        x = x.unsqueeze(2).unsqueeze(-1)           
        u_hat = torch.matmul(W, x).squeeze(-1)     
        b = torch.zeros(B, self.in_capsules, self.out_capsules, 1, device=x.device)
        for i in range(self.num_routes):
            c = F.softmax(b, dim=2)                # [B,Cin,Cout,1]
            s = (c * u_hat).sum(1, keepdim=True)   # [B,1,Cout,Dout]
            v = self.squash(s)                     # [B,1,Cout,Dout]
            if i < self.num_routes - 1:
                b = b + (u_hat * v).sum(-1, keepdim=True)
        return v.squeeze(1)                        # [B,Cout,Dout]
    
class HybridCNN_CapsDepth(pl.LightningModule):
    def __init__(
        self,
        n_classes=4,
        num_caps_layers=1,
        cap_dim=16,
        routing_iters=3,
        dropout_p=0.5,
        focal_alpha=1.0,
        focal_gamma=2.0,
        focal_weight=(1.0, 7.0, 1.0, 2.0),
        use_layernorm=True,
        lr=1e-4,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- ResNet backbone ---
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out_features = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()

        # --- Capsule stack ---
        self.in_capsules = 64
        self.in_dim      = 32
        self.out_caps    = 10
        self.cap_dim     = cap_dim

        self.fc_transform = nn.Linear(self.resnet_out_features, self.in_capsules*self.in_dim)

        caps = []
        caps.append(CapsuleLayer(self.in_capsules, self.in_dim, self.out_caps, self.cap_dim, num_routes=routing_iters))
        for _ in range(max(0, num_caps_layers-1)):
            caps.append(CapsuleLayer(self.out_caps, self.cap_dim, self.out_caps, self.cap_dim, num_routes=routing_iters))
        self.capsule_layers = nn.ModuleList(caps)

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.caps_ln = nn.LayerNorm(self.cap_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,self.cap_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=self.cap_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.resnet_out_features + self.cap_dim, 512)
        self.fc2 = nn.Linear(512, n_classes)

        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=torch.tensor(focal_weight))
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = nn.ModuleDict({
            "val_precision": MulticlassPrecision(num_classes=n_classes, average="macro").to(dev),
            "val_recall":    MulticlassRecall   (num_classes=n_classes, average="macro").to(dev),
            "val_f1":        MulticlassF1Score  (num_classes=n_classes, average="macro").to(dev),
        })
        self.validation_preds, self.validation_labels = [], []

    def forward(self, x):
        res = self.resnet(x)                                  # [B,2048]
        t = self.fc_transform(res).view(x.size(0), self.in_capsules, self.in_dim)

        caps = t
        for layer in self.capsule_layers:
            caps = layer(caps)                                 # [B,10,cap_dim]

        if self.use_layernorm:
            caps = self.caps_ln(caps)

        B = caps.size(0)
        cls = self.cls_token.expand(B,1,self.cap_dim)
        seq = torch.cat([cls, caps], dim=1)                    # [B,11,cap_dim]
        tr  = self.transformer_encoder(seq)
        cls_out = tr[:,0,:]                                    # [B,cap_dim]

        fused = torch.cat([res, cls_out], dim=1)
        fused = self.dropout(fused)
        h = F.relu(self.fc1(fused))
        logits = self.fc2(h)
        return logits

    def training_step(self, batch, _):
        x, y = batch
        if isinstance(self.criterion.w, torch.Tensor) and self.criterion.w.device != x.device:
            self.criterion.w = self.criterion.w.to(x.device)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(1)
        self.validation_preds.append(preds.detach().cpu())
        self.validation_labels.append(y.detach().cpu())
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        f1  = self.metrics["val_f1"](preds, y)
        pre = self.metrics["val_precision"](preds, y)
        self.log("val_f1_step",  f1,  prog_bar=True, on_step=True)
        self.log("val_pre_step", pre, prog_bar=True, on_step=True)

    def on_validation_epoch_end(self):
        if not self.validation_preds: return
        y_pred = torch.cat(self.validation_preds).to(self.device)
        y_true = torch.cat(self.validation_labels).to(self.device)
        f1  = self.metrics["val_f1"](y_pred, y_true)
        pre = self.metrics["val_precision"](y_pred, y_true)
        rec = self.metrics["val_recall"](y_pred, y_true)
        acc = (y_pred == y_true).float().mean()

        self.log("val_f1",  f1,  prog_bar=True, on_epoch=True)
        self.log("val_pre", pre, prog_bar=True, on_epoch=True)
        self.log("val_rec", rec, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        self.validation_preds.clear()
        self.validation_labels.clear()


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return [opt], [sch]


def train_one_caps_depth(dm, num_caps_layers, cap_dim=16, routing_iters=3,
                         max_epochs=30, precision="32-true", tag=None):
    """Trenuje jeden wariant i zwraca słownik metryk (macro) + czasy."""
    model = HybridCNN_CapsDepth(
        n_classes=4,
        num_caps_layers=num_caps_layers,
        cap_dim=cap_dim,
        routing_iters=routing_iters,
        dropout_p=0.5,
        use_layernorm=True,
        lr=1e-4, weight_decay=1e-4,
    )

    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/capsL{num_caps_layers}_dim{cap_dim}_r{routing_iters}",
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, verbose=False
    )
    logger = TensorBoardLogger("tb_logs", name=f"capsL{num_caps_layers}_dim{cap_dim}_r{routing_iters}" + (f"_{tag}" if tag else ""))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=[ckpt, TQDMProgressBar()],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    val_metrics = trainer.validate(model, datamodule=dm)
    t1 = time.time()

    val_log = val_metrics[0] if len(val_metrics) else {}
    acc = float(val_log.get("val_acc", 0.0))
    pre = float(val_log.get("val_pre", 0.0))
    rec = float(val_log.get("val_rec", 0.0))
    f1  = float(val_log.get("val_f1", 0.0))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "num_caps_layers": num_caps_layers,
        "cap_dim": cap_dim,
        "routing_iters": routing_iters,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1": f1,
        "AUC": float("nan"),   # (opcjonalnie policz AUC gdy logujesz proby)
        "train_time_min": (t1 - t0)/60.0,
        "params_M": params/1e6,
    }

    print(f"LaTeX row: {num_caps_layers} warstwa(y) & {acc:.3f} & {pre:.3f} & {rec:.3f} & {f1:.3f} & - \\\\")
    return result


if __name__ == "__main__":
    from pathlib import Path
    torch.set_float32_matmul_precision("high")
    import multiprocessing as mp; mp.freeze_support()

    DATASET_PATH = r"F:/Badania/embedded/Dataset"
    dm = CustomDataModule(DATASET_PATH) 

    configs = [
        dict(num_caps_layers=1, cap_dim=16, routing_iters=3, max_epochs=50, precision="32-true", tag="base"),
        dict(num_caps_layers=2, cap_dim=16, routing_iters=2, max_epochs=50, precision="32-true", tag="2caps_r2"),
        dict(num_caps_layers=1, cap_dim=24, routing_iters=3, max_epochs=50, precision="32-true", tag="capdim24"),
    ]

    rows = []
    for cfg in configs:
        out = train_one_caps_depth(dm, **cfg)
        out["tag"] = cfg.get("tag", "")
        rows.append(out)

    df = pd.DataFrame(rows)
    csv_path = "ablation_caps_depth.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] zapisano: {csv_path}\n")
    print(df.to_string(index=False))
