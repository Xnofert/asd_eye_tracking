from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def build_sample_list(data_dir: str) -> list[tuple[str, str, int]]:
    """扫描 Saliency4ASD 数据集：共享图片目录 + 按类别分开的热力图目录。

    目录结构:
      data_dir/Images/{id}.png          — 原始图片 (ASD 和 TD 共用)
      data_dir/ASD_FixMaps/{id}_s.png   — ASD 组注视热力图
      data_dir/TD_FixMaps/{id}_s.png    — TD 组注视热力图

    每张图片产生两个样本 (ASD=1, TD=0)，按热力图目录中实际存在的文件配对。
    """
    root = Path(data_dir)
    images_dir = root / "Images"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {images_dir}")

    samples: list[tuple[str, str, int]] = []

    for class_name, label in [("ASD", 1), ("TD", 0)]:
        heatmaps_dir = root / f"{class_name}_FixMaps"
        if not heatmaps_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {heatmaps_dir}")

        hm_by_id: dict[str, Path] = {}
        for p in heatmaps_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                stem = p.stem.removesuffix("_s")
                hm_by_id[stem] = p

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            hm_path = hm_by_id.get(img_path.stem)
            if hm_path is None:
                continue
            samples.append((str(img_path), str(hm_path), label))

    return samples


class LinkedTransform:
    """联动数据增强：对 RGB 和热力图施加相同的几何变换，保持空间对齐。
    使用 torchvision.transforms.functional 手动传入共享随机参数，
    而非依赖 RNG 状态同步，确保两路变换严格一致。
    """

    def __init__(self, train: bool = True, data_cfg: dict | None = None, aug_cfg: dict | None = None):
        self.train = train
        # ImageNet 预训练归一化参数
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

        # 从配置读取，缺省兜底
        data_cfg = data_cfg or {}
        aug_cfg = aug_cfg or {}
        self.image_size = data_cfg.get("image_size", 224)
        self.resize_size = data_cfg.get("resize_size", 256)
        self.flip_prob = aug_cfg.get("flip_prob", 0.5)
        self.rotation_range = aug_cfg.get("rotation_range", 15)
        self.crop_scale = tuple(aug_cfg.get("crop_scale", [0.8, 1.0]))
        self.crop_ratio = tuple(aug_cfg.get("crop_ratio", [0.9, 1.1]))
        self.brightness_range = tuple(aug_cfg.get("brightness_range", [0.8, 1.2]))

    def __call__(
        self, rgb: Image.Image, heatmap: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            # 先放大到 resize_size 留出裁剪余量
            rgb = TF.resize(rgb, self.resize_size, interpolation=InterpolationMode.BILINEAR)
            heatmap = TF.resize(heatmap, self.resize_size, interpolation=InterpolationMode.BILINEAR)

            # 随机水平翻转（共享同一随机数）
            if random.random() < self.flip_prob:
                rgb = TF.hflip(rgb)
                heatmap = TF.hflip(heatmap)

            # 随机旋转 ±rotation_range°（共享同一角度）
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            rgb = TF.rotate(rgb, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            heatmap = TF.rotate(
                heatmap, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )

            # 随机裁剪缩放到 image_size（共享同一裁剪框）
            crop_params = RandomResizedCrop.get_params(
                rgb, scale=self.crop_scale, ratio=self.crop_ratio
            )
            size = [self.image_size, self.image_size]
            rgb = TF.resized_crop(rgb, *crop_params, size)
            heatmap = TF.resized_crop(heatmap, *crop_params, size)

            # 仅对 RGB 做亮度扰动，禁止扰动热力图密度分布
            rgb = TF.adjust_brightness(rgb, random.uniform(*self.brightness_range))
        else:
            # 验证时仅做确定性 resize，无随机增强
            size = [self.image_size, self.image_size]
            rgb = TF.resize(rgb, size)
            heatmap = TF.resize(heatmap, size)

        # PIL → Tensor，值域 [0, 1]
        rgb_t = TF.to_tensor(rgb)       # (3, 224, 224)
        hm_t = TF.to_tensor(heatmap)    # (1, 224, 224)

        # 仅对 RGB 做 ImageNet 标准归一化；热力图保持 [0,1] 密度值
        rgb_t = TF.normalize(rgb_t, self.rgb_mean, self.rgb_std)

        return rgb_t, hm_t


class ASDDataset(Dataset):
    """ASD 眼动双模态数据集，返回配对的 RGB 图像、热力图和标签。"""

    def __init__(
        self,
        samples: list[tuple[str, str, int]],
        train: bool = True,
        data_cfg: dict | None = None,
        aug_cfg: dict | None = None,
    ):
        self.samples = samples
        self.transform = LinkedTransform(train=train, data_cfg=data_cfg, aug_cfg=aug_cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, hm_path, label = self.samples[idx]
        rgb = Image.open(img_path).convert("RGB")
        heatmap = Image.open(hm_path).convert("L")
        rgb_t, hm_t = self.transform(rgb, heatmap)
        return {
            "rgb": rgb_t,
            "heatmap": hm_t,
            "label": torch.tensor(label, dtype=torch.float32),
        }
