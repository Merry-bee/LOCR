"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
from .model_prompt import LOCRConfig, LOCRModel
from .utils.dataset import LOCRDataset
from ._version import __version__

__all__ = [
    "LOCRConfig",
    "LOCRDataset",
    "LOCRModel",
]
