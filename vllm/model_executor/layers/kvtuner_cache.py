# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
import torch
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# KVTuner integration requires the KVTuner package
try:
    # Add KVTuner path to sys.path if not already present
    kvtuner_path = "/home/data/so2/KVTuner"
    if kvtuner_path not in sys.path:
        sys.path.insert(0, kvtuner_path)
    
    from flexible_quant.flexible_quant.flexible_quantized_cache import (
        FlexibleQuantizedCacheConfig, 
        FlexibleQuantizedCache,
        FlexibleVanillaQuantizedCache,
        FlexibleQuantoQuantizedCache,
        FlexibleHQQQuantizedCache,
    )
    KVTUNER_AVAILABLE = True
    logger.info("KVTuner integration enabled")
except ImportError as e:
    logger.warning(f"KVTuner not available: {e}. Using standard KV cache.")
    FlexibleQuantizedCacheConfig = None
    FlexibleQuantizedCache = None
    FlexibleVanillaQuantizedCache = None
    FlexibleQuantoQuantizedCache = None
    FlexibleHQQQuantizedCache = None
    KVTUNER_AVAILABLE = False


class KVTunerCacheManager:
    """Manager for KVTuner quantized cache integration with vLLM."""
    
    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        
        # Initialize KVTuner cache if quantization is enabled
        self.kvtuner_cache = None
        self.kvtuner_cache_config = None
        
        if (self.model_config.quantization == "kvtuner" and 
            KVTUNER_AVAILABLE):
            self._initialize_kvtuner_cache()
    
    def _initialize_kvtuner_cache(self):
        """Initialize KVTuner quantized cache."""
        try:
            # Create KVTuner cache configuration
            cache_config_kwargs = {
                "backend": self.cache_config.kvtuner_backend,
                "nbits": -1,  # Use per-layer config
                "nbits_key": -1,
                "nbits_value": -1,
                "axis_key": 0,  # Per-token quantization
                "axis_value": 0,
                "asym": False,
                "q_group_size": -1,
                "residual_length": 0,
                "compute_dtype": torch.float16,
                "device": str(self.device),
                "force_quant": False,
                "per_layer_quant": True,
                "per_layer_config": None,
            }
            
            # Load per-layer configuration if provided
            if self.cache_config.kvtuner_config_path:
                import yaml
                try:
                    with open(self.cache_config.kvtuner_config_path, 'r') as f:
                        per_layer_config = yaml.safe_load(f)
                    cache_config_kwargs["per_layer_config"] = per_layer_config
                    logger.info(f"Loaded KVTuner config from {self.cache_config.kvtuner_config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load KVTuner config: {e}")
            
            self.kvtuner_cache_config = FlexibleQuantizedCacheConfig(**cache_config_kwargs)
            
            # Create appropriate cache based on backend
            backend = self.cache_config.kvtuner_backend
            if backend == "vanilla":
                self.kvtuner_cache = FlexibleVanillaQuantizedCache(self.kvtuner_cache_config)
            elif backend == "quanto":
                self.kvtuner_cache = FlexibleQuantoQuantizedCache(self.kvtuner_cache_config)
            elif backend == "hqq":
                self.kvtuner_cache = FlexibleHQQQuantizedCache(self.kvtuner_cache_config)
            else:
                logger.warning(f"Unknown KVTuner backend: {backend}, using vanilla")
                self.kvtuner_cache = FlexibleVanillaQuantizedCache(self.kvtuner_cache_config)
                
            logger.info(f"Initialized KVTuner cache with backend: {backend}")
            
        except Exception as e:
            logger.error(f"Failed to initialize KVTuner cache: {e}")
            self.kvtuner_cache = None
            self.kvtuner_cache_config = None
    
    def is_quantized(self) -> bool:
        """Check if KVTuner quantization is enabled."""
        return (self.model_config.quantization == "kvtuner" and 
                self.kvtuner_cache is not None)
    
    def update_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with quantization if enabled."""
        if not self.is_quantized():
            # Return unmodified tensors if quantization not enabled
            return key_states, value_states
        
        try:
            # Update the KVTuner cache
            self.kvtuner_cache.update(
                key_states=key_states,
                value_states=value_states,
                layer_idx=layer_idx,
                cache_kwargs=cache_kwargs or {},
            )
            
            # Return the quantized cached states
            cached_keys = self.kvtuner_cache.key_cache[layer_idx] if hasattr(self.kvtuner_cache, 'key_cache') else key_states
            cached_values = self.kvtuner_cache.value_cache[layer_idx] if hasattr(self.kvtuner_cache, 'value_cache') else value_states
            
            return cached_keys, cached_values
            
        except Exception as e:
            logger.error(f"KVTuner cache update failed for layer {layer_idx}: {e}")
            # Fallback to standard behavior
            return key_states, value_states
    
    def get_cache_shape(
        self, 
        num_blocks: int, 
        block_size: int, 
        num_kv_heads: int, 
        head_size: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Get cache shape, accounting for quantization."""
        # For KVTuner, we use the same shape as standard cache
        # The quantization is applied during computation
        key_cache_shape = (num_blocks, num_kv_heads, head_size // 2, block_size, 2)
        value_cache_shape = (num_blocks, num_kv_heads, head_size, block_size)
        return key_cache_shape, value_cache_shape
    
    def get_cache_dtype(self, model_dtype: torch.dtype) -> torch.dtype:
        """Get cache dtype, accounting for quantization."""
        if self.is_quantized():
            # KVTuner handles quantization internally, use model dtype for cache
            return model_dtype
        return model_dtype


def create_kvtuner_cache_manager(vllm_config: "VllmConfig") -> KVTunerCacheManager:
    """Factory function to create KVTuner cache manager."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return KVTunerCacheManager(vllm_config, device)
