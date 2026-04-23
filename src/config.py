from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件，返回嵌套字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
