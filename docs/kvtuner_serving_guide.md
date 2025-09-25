# KVTuner Integration with vLLM

This guide explains how to serve models using KVTuner quantization in vLLM. KVTuner provides calibrated KV cache quantization that reduces memory usage while maintaining model quality through layer-specific bit precision optimization.

## Overview

KVTuner is a research-oriented KV cache quantization framework that:
- Uses **calibrated bit-precision configurations** per layer
- Supports **KIVI** and **per-token** quantization methods
- Provides **model-specific presets** for optimal quality-memory trade-offs
- Achieves **2-8x memory reduction** depending on configuration

## Prerequisites

1. **vLLM with KVTuner Integration**: Ensure you're using the vLLM version with KVTuner support
2. **KVTuner Preset Files**: Calibrated YAML files for your specific model
3. **Compatible GPU**: NVIDIA RTX 4090 or newer (compute capability 7.0+)

## Installation

### Clone and Install vLLM with KVTuner Support

```bash
git clone https://github.com/YourRepo/vllm.git
cd vllm
git checkout kvt2  # Branch with KVTuner integration
pip install -e .
```

### Obtain KVTuner Presets

KVTuner presets are available from the [KVTuner project](https://github.com/YourRepo/KVTuner):

```bash
git clone https://github.com/YourRepo/KVTuner.git
ls KVTuner/calibration_presets/
```

Available presets include:
- `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` (4-bit KIVI)
- `Qwen2.5-3B-Instruct_kivi_KVTuner6_0.yaml` (6-bit KIVI)
- `Qwen2.5-7B-Instruct_pertoken_KVTuner4_0.yaml` (4-bit per-token)
- `Meta-Llama-3.1-8B-Instruct_kivi_KVTuner4_1.yaml`
- And more...

## Basic Usage

### Simple KVTuner Serving

```bash
vllm serve Qwen2.5-3B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path ./Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml \
  --kvtuner-method kivi \
  --host 0.0.0.0 \
  --port 8000
```

### Production Configuration

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen2.5-3B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path /path/to/Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml \
  --kvtuner-method kivi \
  --host 0.0.0.0 \
  --port 8007 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 1024 \
  --enable-prefix-caching \
  --disable-log-requests \
  --uvicorn-log-level error
```

## Configuration Options

### KVTuner-Specific Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--quantization` | Quantization method | - | `kvtuner` |
| `--kvtuner-preset-path` | Path to KVTuner YAML preset | `None` | Path to `.yaml` file |
| `--kvtuner-method` | Quantization algorithm | `kivi` | `kivi`, `pertoken` |

### Quantization Methods

#### KIVI (Key-Value Importance)
- **Best for**: General use cases
- **Features**: Identifies important tokens and preserves them in higher precision
- **Usage**: `--kvtuner-method kivi`

#### Per-Token Quantization
- **Best for**: Fine-grained control
- **Features**: Individual quantization parameters per token
- **Usage**: `--kvtuner-method pertoken`

## Preset File Format

KVTuner preset files are YAML files that specify bit precision for each layer:

```yaml
# Example: Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml
0:  # Layer 0
  nbits_key: 8    # 8-bit key cache
  nbits_value: 8  # 8-bit value cache
1:  # Layer 1
  nbits_key: 2    # 2-bit key cache (aggressive quantization)
  nbits_value: 4  # 4-bit value cache
2:  # Layer 2
  nbits_key: 8    # Back to 8-bit (important layer)
  nbits_value: 8
# ... continues for all layers
```

### Bit Precision Levels

- **8-bit**: Minimal quantization, preserves quality
- **6-bit**: Light quantization, good quality-memory trade-off
- **4-bit**: Moderate quantization, noticeable memory savings
- **2-bit**: Aggressive quantization, maximum memory savings

## Model Examples

### Qwen2.5 Models

```bash
# Qwen2.5-3B with 4-bit KIVI quantization
vllm serve Qwen2.5-3B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml \
  --kvtuner-method kivi

# Qwen2.5-7B with 6-bit per-token quantization
vllm serve Qwen2.5-7B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path Qwen2.5-7B-Instruct_pertoken_KVTuner6_1.yaml \
  --kvtuner-method pertoken
```

### Meta-Llama Models

```bash
# Llama-3.1-8B with KIVI quantization
vllm serve Meta-Llama-3.1-8B-Instruct \
  --quantization kvtuner \
  --kvtuner-preset-path Meta-Llama-3.1-8B-Instruct_kivi_KVTuner4_0.yaml \
  --kvtuner-method kivi \
  --tensor-parallel-size 2
```

### Mistral Models

```bash
# Mistral-7B with per-token quantization
vllm serve Mistral-7B-Instruct-v0.3 \
  --quantization kvtuner \
  --kvtuner-preset-path Mistral-7B-Instruct-v0.3_pertoken_KVTuner4_1.yaml \
  --kvtuner-method pertoken
```

## Testing the API

### OpenAI-Compatible API

Once the server is running, test with:

```bash
curl http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3B-Instruct",
    "messages": [
      {
        "role": "user", 
        "content": "Explain quantum computing in simple terms."
      }
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Python Client

```python
import openai

# Point to your vLLM server
client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8007/v1"
)

response = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! How does KVTuner quantization work?"}
    ],
    max_tokens=150
)

print(response.choices[0].message.content)
```

## Server Logs and Monitoring

### Expected Initialization Logs

```
INFO: Loaded KVTuner preset from /path/to/preset.yaml: method=kivi, 28 layers configured
INFO: KVTuner Layer 0: K=8bit [-128, 127], V=8bit [-128, 127], method=kivi
INFO: KVTuner Layer 1: K=2bit [-2, 1], V=4bit [-8, 7], method=kivi
INFO: KVTuner Layer 2: K=8bit [-128, 127], V=8bit [-128, 127], method=kivi
...
INFO: KVTuner Layer 0 quantization setup complete: K=8bit, V=8bit
INFO: KVTuner Layer 1 quantization setup complete: K=2bit, V=4bit
```

### Memory Usage Monitoring

Monitor GPU memory to see quantization benefits:

```bash
# Before serving (baseline)
nvidia-smi

# After serving with KVTuner
nvidia-smi

# Expected: Reduced memory usage compared to unquantized serving
```

## Performance Considerations

### Memory Savings

| Configuration | Memory Reduction | Quality Impact |
|---------------|------------------|----------------|
| 8-bit uniform | 2x | Minimal |
| 6-bit mixed | 2.5-3x | Light |
| 4-bit mixed | 3-4x | Moderate |
| 2-bit aggressive | 6-8x | Noticeable |

### Latency Impact

- **Quantization overhead**: 2-5% additional latency
- **Memory bandwidth savings**: Can improve overall throughput
- **Cache efficiency**: Better cache locality due to smaller tensors

## Troubleshooting

### Common Issues

#### 1. Preset File Not Found
```bash
# Error: KVTuner preset file not found
# Solution: Check file path and permissions
ls -la /path/to/your/preset.yaml
```

#### 2. YAML Parsing Error
```bash
# Validate YAML syntax
python -c "import yaml; print(yaml.safe_load(open('preset.yaml')))"
```

#### 3. Import Errors
```bash
# Verify KVTuner integration is installed
python -c "from vllm.model_executor.layers.quantization.kvtuner import KVTunerConfig; print('KVTuner OK')"
```

#### 4. GPU Compatibility
```bash
# Check GPU compute capability (should be 7.0+)
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Debug Mode

Enable detailed logging:

```bash
vllm serve model \
  --quantization kvtuner \
  --kvtuner-preset-path preset.yaml \
  --uvicorn-log-level debug
```

## Advanced Usage

### Custom Preset Creation

Create custom presets for your models:

```yaml
# custom_preset.yaml
0: {nbits_key: 8, nbits_value: 8}  # Critical first layer
1: {nbits_key: 4, nbits_value: 6}  # Moderate quantization
2: {nbits_key: 2, nbits_value: 4}  # Aggressive quantization
# ... customize per your needs
```

### Multi-GPU Serving

KVTuner works with tensor parallelism:

```bash
# Distribute across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve model \
  --quantization kvtuner \
  --kvtuner-preset-path preset.yaml \
  --tensor-parallel-size 4
```

### Benchmarking

Compare quantized vs unquantized performance:

```bash
# Baseline (no quantization)
vllm serve model --port 8000

# KVTuner quantized
vllm serve model --quantization kvtuner \
  --kvtuner-preset-path preset.yaml --port 8001

# Use your preferred benchmarking tool to compare
```

## Best Practices

1. **Start with provided presets**: Use calibrated presets for your model family
2. **Monitor quality**: Test with your specific use cases
3. **Profile memory usage**: Verify expected memory savings
4. **Gradual deployment**: Test with non-critical workloads first
5. **Keep backups**: Maintain unquantized model serving as fallback

## Contributing

To add support for new models or improve KVTuner integration:

1. Fork the repository
2. Add new preset files to `calibration_presets/`
3. Test with your model
4. Submit a pull request

## Support

For issues and questions:
- **vLLM Issues**: Create GitHub issues for integration problems
- **KVTuner Issues**: Refer to the KVTuner project repository
- **Documentation**: Check this guide and inline help (`vllm serve --help`)

---

**Note**: KVTuner quantization is an experimental feature. While it provides significant memory savings, thoroughly test with your specific models and use cases before production deployment.