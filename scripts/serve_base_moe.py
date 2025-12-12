import os
import argparse
import subprocess
import sys
import shlex

def main():
    parser = argparse.ArgumentParser(description="Serve Qwen MoE with vLLM (baseline, no code changes).")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat", help="HF model id or local path.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--max-model-len", type=int, default=512, help="Max model length for vLLM engine.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="TP size (set if using multiple GPUs).")
    parser.add_argument("--gpu", action="store_true", help="Force GPU backend if available.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend (slow).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype.")
    parser.add_argument("--download-dir", default=None, help="Optional local model dir to load from.")
    parser.add_argument("--env", action="append", default=[], help="Extra env var KEY=VALUE (repeatable).")
    parser.add_argument("--log-level", default="INFO", help="Server log level.")
    args = parser.parse_args()

    # Ensure precompiled kernels usage
    os.environ.setdefault("VLLM_USE_PRECOMPILED_KERNELS", "1")

    # Apply extra env vars
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v

    # Backend selection
    if args.cpu:
        backend_flag = "--device cpu"
    elif args.gpu:
        backend_flag = "--device cuda"
    else:
        # Auto: cuda if available, else cpu
        backend_flag = "--device auto"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--log-level", args.log_level,
    ]

    if args.download_dir:
        cmd += ["--download-dir", args.download_dir]

    if args.tensor_parallel_size:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]

    # Add backend flag tokens
    cmd += shlex.split(backend_flag)

    print("Starting vLLM OpenAI-compatible server...")
    print("Command:", " ".join(shlex.quote(c) for c in cmd))
    print("Env: VLLM_USE_PRECOMPILED_KERNELS=", os.environ.get("VLLM_USE_PRECOMPILED_KERNELS", ""))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Server exited with error:", e)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()