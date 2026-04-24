from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import ASDDataset, build_sample_list
from src.model import TwoStreamNet


def compute_class_weight(labels: list[int]) -> torch.Tensor:
    """计算 BCEWithLogitsLoss 的 pos_weight，用于处理类别不平衡。
    pos_weight = 负样本数 / 正样本数，使正类损失贡献与负类等量。
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """计算三项核心指标：AUC-ROC、灵敏度 (Sensitivity)、特异性 (Specificity)。"""
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # 正类召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # 负类召回率
    return {"auc_roc": auc, "sensitivity": sensitivity, "specificity": specificity}


class Trainer:
    """训练管理器：封装分层交叉验证、冻结/解冻调度、模型评估与 checkpoint 保存。"""

    def __init__(self, cfg: dict):
        train_cfg = cfg.get("train", {})
        self.data_cfg = cfg.get("data", {})
        self.aug_cfg = cfg.get("augmentation", {})
        self.model_cfg = cfg.get("model", {})

        self.n_folds = train_cfg.get("n_folds", 5)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.lr = train_cfg.get("lr", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-2)
        self.num_epochs = train_cfg.get("num_epochs", 30)
        self.freeze_epochs = train_cfg.get("freeze_epochs", 10)
        self.output_dir = train_cfg.get("output_dir", "outputs")
        self.num_workers = train_cfg.get("num_workers", 4)
        self.random_seed = train_cfg.get("random_seed", 42)

        # 自动检测可用设备：CUDA → MPS (Apple Silicon) → CPU
        device = train_cfg.get("device")
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        data_dir = self.data_cfg.get("data_dir", "data")
        self.samples = build_sample_list(data_dir)
        self.labels = [s[2] for s in self.samples]
        self.groups = [s[3] for s in self.samples]

        n_asd = sum(self.labels)
        n_td = len(self.labels) - n_asd
        n_images = len(set(self.groups))
        print(f"Dataset: {len(self.samples)} samples (ASD={n_asd}, TD={n_td}), {n_images} unique images")
        print(f"Device: {self.device}")

    def run(self) -> list[dict[str, float]]:
        """执行按图像分组的 K 折交叉验证，每折使用全新模型。返回各折最佳验证指标。"""
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            gkf.split(self.samples, self.labels, groups=self.groups)
        ):
            print(f"\n{'=' * 60}")
            print(f"Fold {fold_idx + 1}/{self.n_folds}")
            print(f"{'=' * 60}")

            train_samples = [self.samples[i] for i in train_idx]
            val_samples = [self.samples[i] for i in val_idx]
            train_labels = [self.labels[i] for i in train_idx]

            train_ds = ASDDataset(train_samples, train=True, data_cfg=self.data_cfg, aug_cfg=self.aug_cfg)
            val_ds = ASDDataset(val_samples, train=False, data_cfg=self.data_cfg, aug_cfg=self.aug_cfg)

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

            model = TwoStreamNet(model_cfg=self.model_cfg).to(self.device)
            pos_weight = compute_class_weight(train_labels).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_metrics = self._train_fold(
                model, train_loader, val_loader, criterion, fold_idx
            )
            fold_results.append(best_metrics)

        return fold_results

    def _train_fold(
        self,
        model: TwoStreamNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        fold_idx: int,
    ) -> dict[str, float]:
        """单折训练循环：前 freeze_epochs 冻结 ResNet，之后解冻微调。
        以验证 AUC 为基准保存最佳 checkpoint。
        """
        # 冻结 ResNet-50，先仅训练 GazeCNN + 分类头
        model.freeze_branch_rgb()
        optimizer = self._make_optimizer(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs - self.freeze_epochs)

        best_auc = 0.0
        best_metrics: dict[str, float] = {}

        for epoch in range(self.num_epochs):
            if epoch == self.freeze_epochs:
                model.unfreeze_branch_rgb()
                optimizer = self._make_optimizer(model)
                scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs - self.freeze_epochs)
                print(f"  >> Unfroze RGB branch at epoch {epoch + 1}")

            train_loss = self._train_one_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            val_metrics, val_loss = self._evaluate(model, val_loader, criterion)

            print(
                f"  Epoch {epoch + 1:>3d}/{self.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"AUC={val_metrics['auc_roc']:.4f} | "
                f"Sens={val_metrics['sensitivity']:.4f} | "
                f"Spec={val_metrics['specificity']:.4f}"
            )

            if val_metrics["auc_roc"] > best_auc:
                best_auc = val_metrics["auc_roc"]
                best_metrics = val_metrics.copy()
                self._save_checkpoint(model, fold_idx)

        return best_metrics

    def _make_optimizer(self, model: TwoStreamNet) -> torch.optim.Optimizer:
        """创建 AdamW 优化器，仅包含当前 requires_grad=True 的参数。"""
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _train_one_epoch(
        self,
        model: TwoStreamNet,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """执行一个 epoch 的训练，返回平均损失。"""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            rgb = batch["rgb"].to(self.device)
            heatmap = batch["heatmap"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(rgb, heatmap)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(
        self,
        model: TwoStreamNet,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[dict[str, float], float]:
        """在验证集上评估模型，返回 (指标字典, 平均损失)。"""
        model.eval()
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            rgb = batch["rgb"].to(self.device)
            heatmap = batch["heatmap"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)

            logits = model(rgb, heatmap)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["label"].numpy())

        y_prob = np.concatenate(all_probs).squeeze()
        y_true = np.concatenate(all_labels)
        metrics = compute_metrics(y_true, y_prob)
        return metrics, total_loss / max(n_batches, 1)

    def _save_checkpoint(self, model: TwoStreamNet, fold_idx: int):
        """保存当前折的最佳模型权重到 outputs/best_foldN.pt。"""
        path = Path(self.output_dir) / f"best_fold{fold_idx}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"  >> Saved checkpoint: {path}")
