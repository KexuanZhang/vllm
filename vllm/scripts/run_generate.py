import json
import os
import random
import time
from vllm import LLM, SamplingParams

random.seed(1234)

log_path = os.environ.get("VLLM_LOG_MOE", "")
logging_enabled = bool(log_path)

# Load prompts
prompts_file = "prompts.txt"
try:
    with open(prompts_file, 'r') as f:
        prompts = f.read().split("\n\n---\n\n")
    print("File loaded successfully")
except FileNotFoundError:
    raise FileNotFoundError("File not found")

# Initialize model
sp = SamplingParams(temperature=0.0, max_tokens=128)
print("Initializing LLM...")
llm = LLM(
    model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    max_model_len=512,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)

print(f"MoE logging status: {logging_enabled}")

# Generation
t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()

tokens_generated = sum(len(o.outputs[0].token_ids) for o in outs)
wall_time = t1 - t0

# Write timing
timing_file = "timing.json"
timing_data = {}
if os.path.exists(timing_file):
    with open(timing_file, "r") as f:
        timing_data = json.load(f)

result = {
    "wall_time_sec": round(wall_time, 3),
    "tokens_generated": tokens_generated,
    "throughput_tokens_per_sec": round(tokens_generated / wall_time, 2),
    "num_prompts": len(prompts),
}

if logging_enabled:
    timing_data["log"] = result
else:
    timing_data["no_log"] = result

with open(timing_file, "w") as f:
    json.dump(timing_data, f, indent=2)

# Print summary
print("Generation Complete!")
print(f"Tokens generated: {tokens_generated}")
print(f"Wall time: {wall_time:.2f}s")
print(f"Throughput: {tokens_generated / wall_time:.2f} tokens")
print(f"Timing results saved to: {timing_file}")
if logging_enabled:
    print(f"MoE routes logged to: {log_path}")

# Some generation results
if len(outs) > 0:
    print(f"Prompt: {prompts[0][:100]}...")
    print(f"Output: {outs[0].outputs[0].text[:200]}...")
