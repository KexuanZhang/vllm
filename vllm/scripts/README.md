# Takehome Task Summary [Kexuan Zhang]

## Code Modification

- The Moe logger is **hooked at /vllm/vllm/model_executor/layers/fused_moe/layer.py** inside method select_experts()
- To complete the main functionalities, three helper functions are implemented in layer.py: init_moe_logger(), wite_header(), log_expert_selection()
- The /vllm/vllm/envs.py is updated to register the env variables for Moe logger: 

  | Variable | Description | Default |
  |----------|-------------|---------|
  | `VLLM_LOG_MOE` | Path to JSONL log file. Empty = disabled | `""` |
  | `VLLM_LOG_MOE_LAYERS` | Comma-separated layer indices to log | `"0"` |

## Commands to Run

### Step 0: All Runnable Scripts are under /vllm/vllm/scripts

### Step 1: Generate Prompts

```bash
python vllm/scripts/make_prompts.py
```

### Step 2: Run without Logging
```bash
unset VLLM_LOG_MOE
python vllm/scripts/run_generate.py
```

### Step 3: Run with Logging

```bash
VLLM_LOG_MOE=moe_routes.jsonl python vllm/scripts/run_generate.py
```

To log a different layer:
```bash
VLLM_LOG_MOE=moe_routes.jsonl VLLM_LOG_MOE_LAYER=5 python vllm/scripts/run_generate.py
```

### Step 4: Plot Expert Usage Histogram

```bash
python vllm/scripts/plot_expert_usage.py moe_routes.jsonl -o expert_hist.png
```

### Step 5: Analyze Expert Statistics
```bash
python vllm/scripts/analyze_expert_usage.py moe_routes.jsonl
```

### Step 6: Read Timing
```bash
cat timing.json
```

## Stats Analysis

- Top 3 experts are:
  1. Expert 5 with 23.7% (23327) selections
  2. Expert 58 with 19.4% (19102) selections
  3. Expert 43 with 19.4% (19053) selections
- Normalized distributions are centered with expert 5, 7, 43, 58 for around 20% selections, then expert 18, 33, 38 for around 4.4%. And the remaining experts are all <0.2%
- The entropy for this case is 2.959 bits
- **Interpretation:** the stats shows a highly skewed distribution with strong specialization, that only 7 out of 60 experts handle ~95% of all tokens, confirming significant load imbalance.

## AI usage log

### Tools Used
- **Claude Opus 4.5**

### Use Cases
1. **Understanding vLLM structure** - Navigating the codebase to locate where MoE expert selection happens (`select_experts()` in `layer.py`)
2. **Debugging** - Identifying issues with tensor operations, CUDA graph capture detection, and file I/O
3. **Small refactoring** - Simplifying logging functions, cleaning up code structure, and writing helper scripts


### Output Verification
- **Debug logging** - Added print statements to trace execution flow
- **Manual inspection** - Verified JSONL output format and schema correctness by examining `moe_routes.jsonl`
- **Tensor shape validation** - Confirmed `topk_ids` shape `(num_tokens, top_k)` matches expected dimensions
- **End-to-end testing** - Ran full pipeline (prompts → inference → logging → plotting) to verify integrations