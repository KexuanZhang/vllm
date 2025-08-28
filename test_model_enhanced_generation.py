#!/usr/bin/env python3
"""
Simple script to test if a model can generate responses using the enhanced vLLM with stats logging.
This script loads a model from a provided path, sends a simple "Hi" prompt,
and prints the output along with performance statistics.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_local_vllm():
    """Add local vLLM to Python path."""
    # Try multiple potential locations
    vllm_path = None
    if 'VLLM_PATH' in os.environ:
        vllm_path = Path(os.environ['VLLM_PATH'])
    
    if vllm_path is None or not vllm_path.exists():
        # Default path within the workspace
        vllm_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "vllm"
        if not vllm_path.exists():
            vllm_path = Path("/Users/zhang/Desktop/huawei/so1/vllm")
    
    if vllm_path.exists():
        logger.info(f"Found vLLM at: {vllm_path}")
        sys.path.insert(0, str(vllm_path))
        return True
    else:
        logger.error("Could not locate vLLM. Set VLLM_PATH environment variable.")
        return False

def test_model_generation_with_stats(model_path):
    """Test if a model can generate responses using enhanced vLLM with stats."""
    try:
        # Add local vLLM to Python path
        if not setup_local_vllm():
            return False, "Failed to setup local vLLM"
        
        # Import vLLM with stats
        from vllm.offline_llm_with_stats import OfflineLLMWithStats
        from vllm import SamplingParams
        
        # Check if CUDA is available
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This script requires GPU.")
            return False, "CUDA not available"
        
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set memory utilization to avoid OOM errors
        gpu_memory_utilization = 0.7  # Use 70% of GPU memory
        
        # Initialize the LLM with stats logging enabled
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"This might take a while depending on the model size...")
        
        start_time = time.time()
        llm = OfflineLLMWithStats(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            log_stats=True,  # Enable stats logging
            log_stats_interval=1
        )
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=100
        )
        
        # Generate a response to "Hi"
        prompt = "Hi"
        logger.info(f"Sending prompt: '{prompt}'")
        
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, log_detailed_stats=True)
        generation_time = time.time() - start_time
        
        # Print the response
        generated_text = outputs[0].outputs[0].text.strip()
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info("-" * 40)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {generated_text}")
        logger.info("-" * 40)
        
        # Get performance stats
        stats = llm.get_current_stats()
        logger.info("Performance Statistics:")
        for key, value in stats.items():
            if key.startswith('engine_'):
                logger.info(f"  {key}: {value}")
        
        # Return success
        return True, generated_text
        
    except Exception as e:
        logger.error(f"Error testing model with stats: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test model generation with enhanced vLLM and stats logging")
    parser.add_argument("model_path", help="Path to the model or model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--disable-stats", action="store_true", help="Disable stats logging")
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES to use the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(f"Using GPU: {args.gpu}")
    
    # Set environment variable for local vLLM path if not already set
    if 'VLLM_PATH' not in os.environ:
        # Default path to local vLLM
        vllm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        os.environ['VLLM_PATH'] = vllm_path
        logger.info(f"Set VLLM_PATH to: {vllm_path}")
    
    # Test model generation with stats
    success, output = test_model_generation_with_stats(args.model_path)
    
    if success:
        logger.info("✅ Model test with stats successful!")
        sys.exit(0)
    else:
        logger.error(f"❌ Model test with stats failed: {output}")
        sys.exit(1)

if __name__ == "__main__":
    main()
