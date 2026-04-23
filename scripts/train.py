"""ASD 眼动筛查模型训练入口。

用法:
    python -m scripts.train                                  # 使用默认 config.yaml
    python -m scripts.train --config my_config.yaml          # 指定配置文件
    python -m scripts.train --batch_size 16 --num_epochs 10  # CLI 覆盖配置文件中的值
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 `from src...` 可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ASD eye-tracking dual-modal classifier"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径 (default: config.yaml)")
    # 以下参数可选，提供时覆盖 config.yaml 中的对应值
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """将命令行显式传入的参数覆盖到 cfg 字典中。"""
    if args.data_dir is not None:
        cfg.setdefault("data", {})["data_dir"] = args.data_dir
    if args.output_dir is not None:
        cfg.setdefault("train", {})["output_dir"] = args.output_dir

    train_overrides = {
        "n_folds": args.n_folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "freeze_epochs": args.freeze_epochs,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    for key, val in train_overrides.items():
        if val is not None:
            cfg.setdefault("train", {})[key] = val

    return cfg


def main():
    """加载配置 → 应用 CLI 覆盖 → 执行交叉验证训练 → 打印指标汇总。"""
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    trainer = Trainer(cfg)
    fold_results = trainer.run()

    import numpy as np

    print(f"\n{'=' * 60}")
    print("Cross-Validation Results")
    print(f"{'=' * 60}")
    for i, metrics in enumerate(fold_results):
        print(
            f"  Fold {i + 1}: AUC={metrics['auc_roc']:.4f}  "
            f"Sens={metrics['sensitivity']:.4f}  "
            f"Spec={metrics['specificity']:.4f}"
        )

    mean_auc = np.mean([m["auc_roc"] for m in fold_results])
    mean_sens = np.mean([m["sensitivity"] for m in fold_results])
    mean_spec = np.mean([m["specificity"] for m in fold_results])
    print(f"  Mean:   AUC={mean_auc:.4f}  Sens={mean_sens:.4f}  Spec={mean_spec:.4f}")


if __name__ == "__main__":
    main()
