"""Test KVTuner integration with vLLM."""

import tempfile
import yaml
import pytest
import torch

from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig, KVTunerMethod


def test_kvtuner_config_creation():
    """Test KVTuner config creation with basic parameters."""
    config = KVTunerConfig(method="kivi")
    assert config.method == "kivi"
    assert config.get_name() == "kvtuner"


def test_kvtuner_config_with_preset():
    """Test KVTuner config creation with YAML preset."""
    # Create a temporary YAML preset file
    preset_data = {
        0: {"nbits_key": 8, "nbits_value": 8},
        1: {"nbits_key": 4, "nbits_value": 6},
        2: {"nbits_key": 2, "nbits_value": 4},
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(preset_data, f)
        preset_path = f.name
    
    try:
        config = KVTunerConfig(preset_path=preset_path, method="kivi")
        assert config.method == "kivi"
        assert len(config.layer_configs) == 3
        assert config.layer_configs[0]['nbits_key'] == 8
        assert config.layer_configs[1]['nbits_value'] == 6
        assert config.layer_configs[2]['nbits_key'] == 2
    finally:
        import os
        os.unlink(preset_path)


def test_kvtuner_method_creation():
    """Test KVTuner method creation and layer configuration."""
    config = KVTunerConfig(method="kivi")
    config.layer_configs = {0: {"nbits_key": 4, "nbits_value": 6}}
    
    method = KVTunerMethod(config, "model.layers.0.self_attn")
    assert method.layer_idx == 0
    assert method.layer_config['nbits_key'] == 4
    assert method.layer_config['nbits_value'] == 6


def test_kvtuner_quantization_functions():
    """Test KVTuner quantization and dequantization functions."""
    import torch.nn as nn
    
    # Create a mock attention layer
    layer = nn.Module()
    
    config = KVTunerConfig(method="kivi")
    config.layer_configs = {0: {"nbits_key": 4, "nbits_value": 6}}
    
    method = KVTunerMethod(config, "model.layers.0.self_attn")
    
    # Create weights and process them
    method.create_weights(layer)
    method.process_weights_after_loading(layer)
    
    # Test quantization
    key_tensor = torch.randn(32, 8, 64)  # [seq_len, num_heads, head_dim]
    value_tensor = torch.randn(32, 8, 64)
    
    # Apply quantization
    quantized_key, k_scale = layer.kvtuner_quantize_key(key_tensor)
    quantized_value, v_scale = layer.kvtuner_quantize_value(value_tensor)
    
    # Check that quantization worked
    assert quantized_key.shape == key_tensor.shape
    assert quantized_value.shape == value_tensor.shape
    assert isinstance(k_scale, float)
    assert isinstance(v_scale, float)
    
    # Test dequantization
    dequantized_key = layer.kvtuner_dequantize(quantized_key, k_scale)
    dequantized_value = layer.kvtuner_dequantize(quantized_value, v_scale)
    
    # Check shapes
    assert dequantized_key.shape == key_tensor.shape
    assert dequantized_value.shape == value_tensor.shape


def test_kvtuner_layer_extraction():
    """Test layer index extraction from different prefixes."""
    config = KVTunerConfig(method="kivi")
    
    # Test different layer prefix formats
    test_cases = [
        ("model.layers.0.self_attn", 0),
        ("model.layers.15.self_attn", 15),
        ("transformer.layers.5.attn", 5),
    ]
    
    for prefix, expected_idx in test_cases:
        method = KVTunerMethod(config, prefix)
        assert method.layer_idx == expected_idx


def test_kvtuner_bit_ranges():
    """Test KVTuner quantization ranges for different bit precisions."""
    import torch.nn as nn
    
    layer = nn.Module()
    config = KVTunerConfig(method="kivi")
    
    # Test different bit configurations
    bit_configs = [
        {"nbits_key": 2, "nbits_value": 2},
        {"nbits_key": 4, "nbits_value": 4},
        {"nbits_key": 6, "nbits_value": 6},
        {"nbits_key": 8, "nbits_value": 8},
    ]
    
    for bit_config in bit_configs:
        config.layer_configs = {0: bit_config}
        method = KVTunerMethod(config, "model.layers.0.self_attn")
        method.create_weights(layer)
        
        k_bits = bit_config["nbits_key"]
        v_bits = bit_config["nbits_value"]
        
        if k_bits > 1:
            expected_k_qmin = -(2 ** (k_bits - 1))
            expected_k_qmax = 2 ** (k_bits - 1) - 1
            assert layer.kvtuner_k_qmin == expected_k_qmin
            assert layer.kvtuner_k_qmax == expected_k_qmax
        
        if v_bits > 1:
            expected_v_qmin = -(2 ** (v_bits - 1))
            expected_v_qmax = 2 ** (v_bits - 1) - 1
            assert layer.kvtuner_v_qmin == expected_v_qmin
            assert layer.kvtuner_v_qmax == expected_v_qmax


if __name__ == "__main__":
    test_kvtuner_config_creation()
    test_kvtuner_config_with_preset()
    test_kvtuner_method_creation()
    test_kvtuner_quantization_functions()
    test_kvtuner_layer_extraction()
    test_kvtuner_bit_ranges()
    print("All KVTuner tests passed!")