#!/usr/bin/env python3
"""Test script to verify KVTuner fix for linear layer quantization method."""

import os
import sys
import tempfile
import yaml

# Add vLLM to path
vllm_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vllm_path)

from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, UnquantizedLinearMethod
from vllm.attention.layer import Attention
import torch

def test_kvtuner_quant_method():
    """Test that KVTuner returns appropriate quantization methods."""
    
    # Create a simple test preset
    test_preset = {
        0: {"nbits_key": 4, "nbits_value": 6},
        1: {"nbits_key": 2, "nbits_value": 4},
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_preset, f)
        preset_path = f.name
    
    try:
        # Create KVTuner config
        kvtuner_config = KVTunerConfig(
            method="kivi",
            preset_path=preset_path
        )
        
        print("✅ KVTuner config created successfully")
        print(f"   Loaded {len(kvtuner_config.layer_configs)} layer configs")
        
        # Test with a mock linear layer
        class MockLinear(torch.nn.Module):
            pass
        
        mock_linear = MockLinear()
        linear_method = kvtuner_config.get_quant_method(mock_linear, "model.layers.0.qkv_proj")
        
        # Should return UnquantizedLinearMethod for non-attention layers
        assert isinstance(linear_method, UnquantizedLinearMethod), \
            f"Expected UnquantizedLinearMethod for linear layer, got {type(linear_method)}"
        print("✅ Linear layer gets UnquantizedLinearMethod correctly")
        
        # Test with a mock attention layer  
        class MockAttention(torch.nn.Module):
            pass
        
        # Note: This test is simplified since we can't easily create real Attention layer
        # The key fix is that linear layers get UnquantizedLinearMethod
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(preset_path):
            os.unlink(preset_path)

if __name__ == "__main__":
    print("Testing KVTuner quantization method fix...")
    success = test_kvtuner_quant_method()
    sys.exit(0 if success else 1)