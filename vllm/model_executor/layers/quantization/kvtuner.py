# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import yaml
from typing import Any, Dict, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

logger = init_logger(__name__)


class KVTunerConfig(QuantizationConfig):
    """Configuration class for KVTuner quantization.
    
    KVTuner provides layer-wise mixed precision KV cache quantization
    with preset configurations for memory-efficient inference.
    """

    def __init__(
        self,
        kvtuner_config_path: Optional[str] = None,
        kvtuner_scheme: str = "per_token",
        kvtuner_backend: str = "vanilla",
        compute_dtype: torch.dtype = torch.float16,
        force_quant: bool = False,
        residual_length: int = 0,
        q_group_size: int = -1,
        axis_key: int = 0,
        axis_value: int = 0,
        asym: bool = False,
    ) -> None:
        """Initialize KVTuner configuration.
        
        Args:
            kvtuner_config_path: Path to KVTuner preset configuration YAML file
            kvtuner_scheme: Quantization scheme ("per_token", "per_channel", etc.)
            kvtuner_backend: Backend to use ("vanilla", "quanto", "hqq")
            compute_dtype: Compute dtype for the model
            force_quant: Whether to quantize during prefill stage
            residual_length: Length of residual tokens not quantized
            q_group_size: Group size for quantization (-1 for per-token)
            axis_key: Axis for key quantization (0: per-token, 1: per-channel)
            axis_value: Axis for value quantization (0: per-token, 1: per-channel)
            asym: Whether to use asymmetric quantization
        """
        self.kvtuner_config_path = kvtuner_config_path
        self.kvtuner_scheme = kvtuner_scheme
        self.kvtuner_backend = kvtuner_backend
        self.compute_dtype = compute_dtype
        self.force_quant = force_quant
        self.residual_length = residual_length
        self.q_group_size = q_group_size
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.asym = asym
        
        # Load per-layer configuration if config path is provided
        self.per_layer_config = {}
        if kvtuner_config_path and os.path.exists(kvtuner_config_path):
            try:
                with open(kvtuner_config_path, 'r') as f:
                    self.per_layer_config = yaml.safe_load(f)
                logger.info(f"Loaded KVTuner config from {kvtuner_config_path}")
                logger.info(f"Per-layer config: {self.per_layer_config}")
            except Exception as e:
                logger.warning(f"Failed to load KVTuner config from {kvtuner_config_path}: {e}")
                self.per_layer_config = {}

    def get_name(self) -> str:
        return "kvtuner"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Minimum compute capability for KVTuner

    def get_config_filenames(self) -> list[str]:
        return ["kvtuner_config.yaml"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KVTunerConfig":
        """Create KVTunerConfig from a dictionary."""
        return cls(**config)

    def get_kv_cache_method(self) -> "KVTunerKVCacheMethod":
        """Get KV cache method for KVTuner quantization."""
        return KVTunerKVCacheMethod(self)


class KVTunerKVCacheMethod(BaseKVCacheMethod):
    """KV cache method for KVTuner quantization."""

    def __init__(self, config: KVTunerConfig):
        super().__init__()
        self.config = config

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Get the shape of the KV cache for KVTuner quantization."""
        # KVTuner uses the same shape as standard cache for now
        # The quantization is applied during computation
        key_cache_shape = (num_blocks, num_kv_heads, head_size // 2, block_size, 2)
        value_cache_shape = (num_blocks, num_kv_heads, head_size, block_size)
        return key_cache_shape, value_cache_shape

    def get_kv_cache_dtype(
        self,
        model_dtype: torch.dtype,
        kv_cache_dtype: Optional[str] = None,
    ) -> torch.dtype:
        """Get the data type of the KV cache for KVTuner quantization."""
        # KVTuner uses quantized storage but this returns the cache dtype
        # The actual quantization is handled in the attention layers
        if kv_cache_dtype == "auto":
            return model_dtype
        elif kv_cache_dtype is not None:
            return getattr(torch, kv_cache_dtype)
        return model_dtype

    def is_kv_cache_quantization_enabled(self) -> bool:
        """Check if KV cache quantization is enabled."""
        return True

    def get_cache_block_size(
        self,
        block_size: int,
        cache_dtype: torch.dtype,
        model_dtype: torch.dtype,
    ) -> int:
        """Get the cache block size for KVTuner quantization."""
        # For KVTuner, we keep the same block size for simplicity
        # The quantization savings come from the mixed precision per layer
        return block_size
