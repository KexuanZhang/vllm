"""Example script to test KVTuner integration with vLLM serving."""

import tempfile
import yaml
import os

def create_sample_preset():
    """Create a sample KVTuner preset file for testing."""
    # Sample preset based on Qwen2.5-3B-Instruct configuration
    preset_data = {}
    
    # Configure different layers with different bit precisions
    # Layers 0-5: 8-bit (high precision for early layers)
    for i in range(6):
        preset_data[i] = {"nbits_key": 8, "nbits_value": 8}
    
    # Layers 6-15: 6-bit (moderate precision for middle layers)  
    for i in range(6, 16):
        preset_data[i] = {"nbits_key": 6, "nbits_value": 6}
    
    # Layers 16-25: 4-bit (aggressive quantization for later layers)
    for i in range(16, 26):
        preset_data[i] = {"nbits_key": 4, "nbits_value": 4}
    
    # Layer 26-27: 2-bit (very aggressive for final layers)
    for i in range(26, 28):
        preset_data[i] = {"nbits_key": 2, "nbits_value": 4}
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='_kvtuner_test.yaml', delete=False) as f:
        yaml.dump(preset_data, f, default_flow_style=False)
        return f.name

def main():
    """Main function to demonstrate KVTuner usage."""
    print("Creating KVTuner test preset...")
    preset_path = create_sample_preset()
    
    try:
        print(f"Created preset file: {preset_path}")
        
        # Display preset contents
        with open(preset_path, 'r') as f:
            print("Preset contents (first 10 lines):")
            lines = f.readlines()
            for line in lines[:10]:
                print(f"  {line.strip()}")
            if len(lines) > 10:
                print(f"  ... and {len(lines) - 10} more lines")
        
        print(f"\nTo test KVTuner with vLLM, use:")
        print(f"vllm serve <model_path> \\")
        print(f"  --quantization kvtuner \\")
        print(f"  --kvtuner-preset-path {preset_path} \\")
        print(f"  --kvtuner-method kivi")
        
        print(f"\nAlternatively, with per-token method:")
        print(f"vllm serve <model_path> \\")
        print(f"  --quantization kvtuner \\")
        print(f"  --kvtuner-preset-path {preset_path} \\")
        print(f"  --kvtuner-method pertoken")
        
        print(f"\nExample with a specific model:")
        print(f"vllm serve microsoft/DialoGPT-medium \\")
        print(f"  --quantization kvtuner \\")
        print(f"  --kvtuner-preset-path {preset_path} \\")
        print(f"  --kvtuner-method kivi \\")
        print(f"  --tensor-parallel-size 1 \\")
        print(f"  --max-model-len 2048")
        
        # Test basic functionality
        print(f"\n" + "="*50)
        print("Testing KVTuner configuration...")
        
        from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig
        
        # Test config creation
        config = KVTunerConfig(preset_path=preset_path, method="kivi")
        print(f"✓ Created KVTuner config with {len(config.layer_configs)} layer configurations")
        print(f"✓ Method: {config.method}")
        
        # Display some layer configurations
        print(f"\nLayer configurations:")
        for i in [0, 10, 20, 27]:
            if i in config.layer_configs:
                layer_config = config.layer_configs[i]
                print(f"  Layer {i}: K={layer_config['nbits_key']}bit, V={layer_config['nbits_value']}bit")
        
        print(f"\n✓ KVTuner integration test completed successfully!")
        print(f"\nPreset file saved at: {preset_path}")
        print(f"You can use this file to test KVTuner with vLLM.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up (comment out if you want to keep the file)
        # os.unlink(preset_path)
        pass

if __name__ == "__main__":
    main()