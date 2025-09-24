import os, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from model_ENGRAP_heatmap import CapsuleLayer, FocalLoss, CustomDataModule

class HybridCNN(pl.LightningModule):
    def __init__(self, n_classes=4, cap_dim=16, routing_iters=3, dropout_p=0.5, 
                 lr=1e-4, weight_decay=1e-4, **kwargs):
        super(HybridCNN, self).__init__()
        

        self.save_hyperparameters()
        

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_out_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.in_capsules = 64
        self.in_dim = 32
        self.out_capsules = 10
        self.out_dim = cap_dim  
        self.cap_dim = cap_dim  
        
        self.fc_transform = nn.Linear(self.resnet_out_features, self.in_capsules * self.in_dim)
        self.capsule_layer = CapsuleLayer(
            in_capsules=self.in_capsules,
            in_dim=self.in_dim,
            out_capsules=self.out_capsules,
            out_dim=self.out_dim,
            num_routes=routing_iters  
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc1 = nn.Linear(self.resnet_out_features + self.out_dim, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout(dropout_p)  # Używamy parametru dropout_p
        
        if n_classes == 4:
            class_weights = torch.tensor([1.0, 7.0, 1.0, 2.0])
        else:
            class_weights = torch.ones(n_classes)  
            
        self.criterion = FocalLoss(alpha=1, gamma=2, weight=class_weights)
        
        self.n_classes = n_classes
        
        self.validation_preds = []
        self.validation_labels = []
        self.validation_probs = []  

    def setup(self, stage=None):
        pass

    def forward(self, x):
        resnet_output = self.resnet(x)  
        transformed_output = self.fc_transform(resnet_output)  
        capsule_input = transformed_output.view(x.size(0), self.in_capsules, self.in_dim)  # [B, 64, 32]

        capsule_output = self.capsule_layer(capsule_input)  

        B = capsule_output.size(0)
        cls_tokens = self.cls_token.expand(B, 1, self.cap_dim)  
        capsule_with_cls = torch.cat((cls_tokens, capsule_output), dim=1) 

        transformer_output = self.transformer_encoder(capsule_with_cls) 
        cls_output = transformer_output[:, 0, :]  

        combined_features = torch.cat((resnet_output, cls_output), dim=1)  
        combined_features = self.dropout(combined_features)
        h = F.relu(self.fc1(combined_features))
        logits = self.fc2(h)  
        return logits

    def training_step(self, batch, _):
        x, y = batch
        if isinstance(self.criterion.w, torch.Tensor) and self.criterion.w.device != x.device:
            self.criterion.w = self.criterion.w.to(x.device)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.validation_preds.append(preds.detach().cpu())
        self.validation_labels.append(y.detach().cpu())
        self.validation_probs.append(probs.detach().cpu())

        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if not self.validation_preds:
            return
        y_pred = torch.cat(self.validation_preds)
        y_true = torch.cat(self.validation_labels)
        y_proba = torch.cat(self.validation_probs)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred_np = y_pred.numpy()
        y_true_np = y_true.numpy()
        
        acc = accuracy_score(y_true_np, y_pred_np)
        pre = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        rec = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        try:
            y_oh = F.one_hot(y_true, num_classes=y_proba.shape[1]).numpy()
            auc_macro = roc_auc_score(y_oh, y_proba.numpy(), average="macro", multi_class="ovr")
        except Exception:
            auc_macro = float("nan")

        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.log('val_pre', pre, prog_bar=True, on_epoch=True)
        self.log('val_rec', rec, prog_bar=True, on_epoch=True)
        self.log('val_f1',  f1,  prog_bar=True, on_epoch=True)
        self.log('val_auc', auc_macro if not math.isnan(auc_macro) else 0.0, prog_bar=True, on_epoch=True)

        self.validation_preds.clear()
        self.validation_labels.clear()
        self.validation_probs.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        return [opt], [sch]

def train_and_eval_variant(dm, *, cap_dim, routing_iters, dropout_p,
                           max_epochs=50, precision="32-true", tag=""):
    model = HybridCNN(
        n_classes=4,
        cap_dim=cap_dim,
        routing_iters=routing_iters,
        dropout_p=dropout_p,
        lr=1e-4, weight_decay=1e-4,
    )

    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/hp_{tag}",
        filename="best-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1", mode="max", save_top_k=1, verbose=False
    )
    logger = TensorBoardLogger("tb_logs", name=f"hp_{tag}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,              # "32-true" stabilniej niż 16-mixed przy kapsułach
        logger=logger,
        callbacks=[ckpt, TQDMProgressBar()],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    val_list = trainer.validate(model, datamodule=dm)
    t1 = time.time()

    val = val_list[0] if val_list else {}
    row = {
        "Group": "",
        "Config": f"cap_dim={cap_dim}, r={routing_iters}, drop={dropout_p:.1f}",
        "Accuracy": float(val.get("val_acc", 0.0)),
        "Precision": float(val.get("val_pre", 0.0)),
        "Recall": float(val.get("val_rec", 0.0)),
        "F1": float(val.get("val_f1", 0.0)),
        "AUC": float(val.get("val_auc", np.nan)),
        "train_time_min": (t1 - t0)/60.0,
    }

    # LaTeX line
    print(f"LaTeX: {row['Config']} & {row['Accuracy']:.3f} & {row['Precision']:.3f} "
          f"& {row['Recall']:.3f} & {row['F1']:.3f} & {row['AUC']:.3f} \\\\")
    return row


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    torch.set_float32_matmul_precision("high")

    DATASET_PATH = r"F:/Badania/embedded/Dataset"
    dm = CustomDataModule(DATASET_PATH)

    results = []

    for r in [1, 3, 5]:
        row = train_and_eval_variant(dm, cap_dim=16, routing_iters=r, dropout_p=0.5,
                                     max_epochs=20, precision="32-true", tag=f"routing_r{r}")
        row["Group"] = "routing"
        results.append(row)

    for d in [8, 16, 24]:
        row = train_and_eval_variant(dm, cap_dim=d, routing_iters=3, dropout_p=0.5,
                                     max_epochs=20, precision="32-true", tag=f"capdim_d{d}")
        row["Group"] = "cap_dim"
        results.append(row)

    for p_drop in [0.3, 0.5]:
        row = train_and_eval_variant(dm, cap_dim=16, routing_iters=3, dropout_p=p_drop,
                                     max_epochs=20, precision="32-true", tag=f"dropout_p{p_drop}")
        row["Group"] = "dropout"
        results.append(row)

    df = pd.DataFrame(results)
    out_csv = "hp_ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Zapisano: {out_csv}")
    print(df.to_string(index=False))
