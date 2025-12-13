from vllm import LLM, SamplingParams
import os, json, time, random

random.seed(1234)

prompts = ["say hello"]

sp = SamplingParams(temperature=0.0, max_tokens=128)

llm = LLM(model="Qwen/Qwen1.5-MoE-A2.7B-Chat", max_model_len=512, tensor_parallel_size=2, gpu_memory_utilization=0.9)

log_experts = os.getenv("VLLM_LOG_EXPERTS", "0") == "1"

t0 = time.time()
outs = llm.generate(prompts, sp)
t1 = time.time()

tokens_generated = sum(len(o.outputs[0].token_ids) for o in outs)

timing_data = {}
if os.path.exists("timing.json"):
    with open("timing.json", "r") as f:
        timing_data = json.load(f)

if log_experts:
    timing_data["log"] = {
        "wall_time_sec": t1 - t0,
        "tokens_generated": tokens_generated
    }
else:
    timing_data["no_log"] = {
        "wall_time_sec": t1 - t0,
        "tokens_generated": tokens_generated
    }

with open("timing.json", "w") as f:
    json.dump(timing_data, f, indent=2)

print(f"Generated {tokens_generated} tokens in {t1 - t0:.2f} seconds")
print(f"Throughput: {tokens_generated / (t1 - t0):.2f} tokens/sec")