# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Dict
import torch
import yaml
from pathlib import Path
import re

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.logger import init_logger

logger = init_logger(__name__)


class KVTunerConfig(QuantizationConfig):
    """Configuration class for KVTuner KV cache quantization."""
    
    def __init__(
        self,
        method: str = "kivi",
        preset_path: Optional[str] = None,
        layer_configs: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> None:
        super().__init__()
        
        if method not in ["kivi", "pertoken"]:
            raise ValueError(f"Unsupported KVTuner method: {method}")
        
        self.method = method
        self.preset_path = preset_path
        self.layer_configs = layer_configs or {}
        
        # Load preset if provided
        if preset_path:
            self._load_preset(preset_path)

    def _load_preset(self, preset_path: str):
        """Load KVTuner calibration preset from YAML file."""
        try:
            preset_file = Path(preset_path)
            if not preset_file.exists():
                raise FileNotFoundError(f"KVTuner preset file not found: {preset_path}")
                
            with open(preset_file, 'r') as f:
                preset_data = yaml.safe_load(f)
            
            # Convert YAML data to layer configs
            # YAML format: {0: {nbits_key: 8, nbits_value: 8}, 1: {...}}
            self.layer_configs = {}
            for layer_idx, config in preset_data.items():
                if isinstance(layer_idx, int):
                    self.layer_configs[layer_idx] = {
                        'nbits_key': config.get('nbits_key', 8),
                        'nbits_value': config.get('nbits_value', 8)
                    }
            
            logger.info(f"Loaded KVTuner preset from {preset_path}: "
                       f"method={self.method}, {len(self.layer_configs)} layers configured")
                       
        except Exception as e:
            raise ValueError(f"Failed to load KVTuner preset from {preset_path}: {e}")

    @classmethod
    def get_name(cls) -> str:
        return "kvtuner"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Support RTX 4090 and above

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["kvtuner_config.yaml"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KVTunerConfig":
        method = cls.get_from_keys_or(config, ["method", "kvtuner_method"], "kivi")
        preset_path = cls.get_from_keys_or(config, ["preset_path", "kvtuner_preset"], None)
        
        return cls(
            method=method,
            preset_path=preset_path
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.attention.layer import Attention
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        
        # KVTuner only applies to attention layers for KV cache quantization
        if isinstance(layer, Attention):
            return KVTunerMethod(self, prefix)
        
        # For all other layers (linear, etc.), use unquantized method
        # KVTuner only quantizes KV cache, not weights
        return UnquantizedLinearMethod()


class KVTunerMethod(QuantizeMethodBase):
    """KVTuner implementation using bit-precision quantization paradigm."""
    
    def __init__(self, quant_config: KVTunerConfig, layer_prefix: str):
        self.quant_config = quant_config
        self.method = quant_config.method
        self.layer_configs = quant_config.layer_configs
        self.layer_idx = self._extract_layer_idx(layer_prefix)
        
        # Get bit configuration for this layer
        self.layer_config = self.layer_configs.get(self.layer_idx, {
            'nbits_key': 8,    # Default to 8-bit if not in preset
            'nbits_value': 8
        })

    def _extract_layer_idx(self, prefix: str) -> int:
        """Extract layer index from prefix like 'model.layers.5.self_attn'"""
        match = re.search(r'layers\.(\d+)', prefix)
        return int(match.group(1)) if match else 0

    def create_weights(self, layer: torch.nn.Module):
        """Create KVTuner-specific parameters."""
        
        # Store the actual calibrated bit configurations
        layer.kvtuner_nbits_key = self.layer_config['nbits_key']
        layer.kvtuner_nbits_value = self.layer_config['nbits_value']
        layer.kvtuner_method = self.method
        layer.kvtuner_layer_idx = self.layer_idx
        
        # Compute quantization parameters based on bit precision
        k_bits = layer.kvtuner_nbits_key
        v_bits = layer.kvtuner_nbits_value
        
        # Set up quantization ranges
        layer.kvtuner_k_qmin = -(2 ** (k_bits - 1)) if k_bits > 1 else 0
        layer.kvtuner_k_qmax = 2 ** (k_bits - 1) - 1 if k_bits > 1 else 1
        layer.kvtuner_v_qmin = -(2 ** (v_bits - 1)) if v_bits > 1 else 0
        layer.kvtuner_v_qmax = 2 ** (v_bits - 1) - 1 if v_bits > 1 else 1
        
        # Mark this layer as using KVTuner
        layer.use_kvtuner = True
        
        logger.info(f"KVTuner Layer {self.layer_idx}: "
                   f"K={k_bits}bit [{layer.kvtuner_k_qmin}, {layer.kvtuner_k_qmax}], "
                   f"V={v_bits}bit [{layer.kvtuner_v_qmin}, {layer.kvtuner_v_qmax}], "
                   f"method={self.method}")

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Set up KVTuner runtime quantization functions."""
        
        k_bits = layer.kvtuner_nbits_key
        v_bits = layer.kvtuner_nbits_value
        
        def quantize_key(tensor: torch.Tensor):
            """Quantize key tensor based on bit precision."""
            if k_bits >= 8:
                return tensor, 1.0  # No quantization for 8+ bits
            
            # Dynamic scaling based on tensor range (KVTuner approach)
            tensor_max = torch.abs(tensor).max()
            if tensor_max == 0:
                return tensor, 1.0
                
            qmax = 2 ** (k_bits - 1) - 1
            scale = tensor_max / qmax
            
            quantized = torch.round(tensor / scale).clamp(
                layer.kvtuner_k_qmin, layer.kvtuner_k_qmax)
            
            return quantized, scale
        
        def quantize_value(tensor: torch.Tensor):
            """Quantize value tensor based on bit precision."""
            if v_bits >= 8:
                return tensor, 1.0  # No quantization for 8+ bits
                
            tensor_max = torch.abs(tensor).max()
            if tensor_max == 0:
                return tensor, 1.0
                
            qmax = 2 ** (v_bits - 1) - 1
            scale = tensor_max / qmax
            
            quantized = torch.round(tensor / scale).clamp(
                layer.kvtuner_v_qmin, layer.kvtuner_v_qmax)
            
            return quantized, scale
        
        def dequantize_tensor(quantized_tensor: torch.Tensor, scale: float):
            """Dequantize tensor using scale."""
            return quantized_tensor.float() * scale
        
        # Attach quantization functions to layer
        layer.kvtuner_quantize_key = quantize_key
        layer.kvtuner_quantize_value = quantize_value
        layer.kvtuner_dequantize = dequantize_tensor
        
        logger.info(f"KVTuner Layer {self.layer_idx} quantization setup complete: "
                   f"K={k_bits}bit, V={v_bits}bit")

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError("KVTuner method should not call apply() - "
                          "quantization is handled in attention backends")

    def apply_kivi_quantization(self, key: torch.Tensor, value: torch.Tensor):
        """Apply KIVI-specific quantization logic."""
        # This can be extended with more sophisticated KIVI algorithms
        # For now, use the basic bit-precision quantization
        return self._apply_basic_quantization(key, value)

    def apply_pertoken_quantization(self, key: torch.Tensor, value: torch.Tensor):
        """Apply per-token quantization logic."""
        # This can be extended with per-token specific algorithms
        return self._apply_basic_quantization(key, value)

    def _apply_basic_quantization(self, key: torch.Tensor, value: torch.Tensor):
        """Basic quantization using bit precision."""
        k_bits = self.layer_config['nbits_key']
        v_bits = self.layer_config['nbits_value']
        
        # Quantize key
        if k_bits < 8:
            k_max = torch.abs(key).max()
            k_scale = k_max / (2 ** (k_bits - 1) - 1) if k_max > 0 else 1.0
            quantized_key = torch.round(key / k_scale).clamp(
                self.layer_config.get('k_qmin', -(2**(k_bits-1))),
                self.layer_config.get('k_qmax', 2**(k_bits-1)-1)
            )
        else:
            quantized_key = key
            k_scale = 1.0
        
        # Quantize value
        if v_bits < 8:
            v_max = torch.abs(value).max()
            v_scale = v_max / (2 ** (v_bits - 1) - 1) if v_max > 0 else 1.0
            quantized_value = torch.round(value / v_scale).clamp(
                self.layer_config.get('v_qmin', -(2**(v_bits-1))),
                self.layer_config.get('v_qmax', 2**(v_bits-1)-1)
            )
        else:
            quantized_value = value
            v_scale = 1.0
        
        return quantized_key, quantized_value, (k_scale, v_scale)