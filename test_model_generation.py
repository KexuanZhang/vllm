#!/usr/bin/env python3
"""
Simple script to test if a model can generate responses using vLLM.
This script loads a model from a provided path, sends a simple "Hi" prompt,
and prints the output.
"""

import argparse
import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_generation(model_path, tensor_parallel_size=1, prompt="Hi", max_tokens=100):
    """Test if a model can generate responses."""
    try:
        # Import vLLM
        logger.info(f"Importing vLLM...")
        from vllm import LLM, SamplingParams
        
        # Check if CUDA is available
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This script requires GPU.")
            sys.exit(1)
        
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set memory utilization to avoid OOM errors
        gpu_memory_utilization = 0.7  # Use 70% of GPU memory
        
        # Initialize the LLM
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using tensor parallel size: {tensor_parallel_size}")
        logger.info(f"This might take a while depending on the model size...")
        
        start_time = time.time()
        llm = LLM(model=model_path, 
                 gpu_memory_utilization=gpu_memory_utilization,
                 tensor_parallel_size=tensor_parallel_size)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Create sampling parameters (beam search with low temperature for deterministic output)
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=max_tokens
        )
        
        # Generate a response to the prompt
        logger.info(f"Sending prompt: '{prompt}'")
        
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # Print the response
        generated_text = outputs[0].outputs[0].text.strip()
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info("-" * 40)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {generated_text}")
        logger.info("-" * 40)
        
        # Return success
        return True, generated_text
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test model generation with vLLM")
    parser.add_argument("model_path", help="Path to the model or model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')")
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU ID to use")
    parser.add_argument("--gpus", type=str, help="Comma-separated list of GPU IDs to use (e.g., '4,6')")
    parser.add_argument("--tensor-parallel-size", type=int, help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--prompt", type=str, default="Hi", help="Test prompt to send to the model")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    # Set GPU configuration
    if args.gpus:
        # Use multiple GPUs specified by --gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Using GPUs: {args.gpus}")
        
        # Set tensor parallel size if not specified
        if not args.tensor_parallel_size:
            args.tensor_parallel_size = len(args.gpus.split(','))
            logger.info(f"Setting tensor parallel size to {args.tensor_parallel_size}")
    else:
        # Use single GPU specified by --gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logger.info(f"Using GPU: {args.gpu}")
        
        # Default tensor parallel size for single GPU
        if not args.tensor_parallel_size:
            args.tensor_parallel_size = 1
    
    # Test model generation
    success, output = test_model_generation(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt=args.prompt,
        max_tokens=args.max_tokens
    )
    
    if success:
        logger.info("✅ Model test successful!")
        sys.exit(0)
    else:
        logger.error(f"❌ Model test failed: {output}")
        sys.exit(1)

if __name__ == "__main__":
    main()
