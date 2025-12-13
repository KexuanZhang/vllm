# MoE Expert Selection Logging for vLLM

This patch adds flag-gated logging of MoE (Mixture of Experts) expert selections to vLLM. When enabled, it records which experts are selected for each token during inference.

## Quick Start

```bash
# 1. Generate prompts from GSM8K (first 25 questions)
cd /workspace/vllm/vllm/scripts
python make_prompts.py

# 2. Run baseline (no logging) to get timing
python run_generate.py

# 3. Run with MoE logging enabled
VLLM_LOG_MOE=moe_routes.jsonl python run_generate.py

# 4. Generate histogram plot
python plot_moe_histogram.py moe_routes.jsonl --output expert_hist.png
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_LOG_MOE` | Path to JSONL log file. Empty = disabled | `""` |
| `VLLM_LOG_MOE_LAYERS` | Comma-separated layer indices to log | `"0"` |

### Examples

```bash
# Log layer 0 only (default)
VLLM_LOG_MOE=/tmp/moe_routes.jsonl python run_generate.py

# Log multiple layers
VLLM_LOG_MOE=/tmp/moe_routes.jsonl VLLM_LOG_MOE_LAYERS=0,1,2 python run_generate.py
```

## JSONL Log Schema

### Meta Header (first line)
```json
{
  "type": "meta",
  "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
  "vllm_version": "0.x.y",
  "torch_version": "2.x.y",
  "device": "cuda",
  "seed": 1234,
  "layers_logged": [0],
  "top_k": 4
}
```

### Per-Token Route Record
```json
{
  "type": "route",
  "req_id": "default",
  "token_idx": 17,
  "layer": 0,
  "topk_ids": [3, 12, 7, 1],
  "topk_weights": [0.35, 0.28, 0.22, 0.15]
}
```

## Hook Location

The logging hook is placed in `FusedMoE.select_experts()` method in:
- File: `vllm/model_executor/layers/fused_moe/layer.py`
- Location: Right after `topk_ids` and `topk_weights` are computed, before EPLB remapping

This captures the logical expert selections before any physical mapping or dtype conversion.

## Files Modified

- `vllm/envs.py` - Added `VLLM_LOG_MOE` and `VLLM_LOG_MOE_LAYERS` environment variables
- `vllm/model_executor/layers/fused_moe/layer.py` - Added logging helper functions and hook in `select_experts()`
- `vllm/scripts/make_prompts.py` - Generate prompts from GSM8K
- `vllm/scripts/run_generate.py` - Run generation with timing
- `vllm/scripts/plot_moe_histogram.py` - Plot expert histogram

## Analysis Output

The plot script generates:
1. **Raw counts histogram** - Shows absolute selection counts per expert
2. **Normalized distribution** - Shows selection probability per expert
3. **Statistics**:
   - Top-3 most selected experts
   - Normalized entropy (1.0 = uniform, 0.0 = single expert dominates)
   - Interpretation of the distribution

## Performance Notes

- Logging is completely disabled when `VLLM_LOG_MOE` is not set
- CUDA graph capture is detected and logging is skipped during capture
- Tensors are moved to CPU asynchronously to minimize overhead
- Only configured layers are logged to reduce I/O

## Example Analysis

For Qwen1.5-MoE-A2.7B-Chat with GSM8K prompts:
- Total experts: 60 (4 selected per token)
- Expected entropy if uniform: ~0.95
- Observed patterns show some expert specialization for math reasoning tasks
