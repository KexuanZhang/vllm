"""
Test script for the modified vLLM with stats logging.
"""
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vllm_stats_logging():
    """Test the modified vLLM stats logging functionality."""
    
    try:
        # Import the wrapper
        from vllm.offline_llm_with_stats import OfflineLLMWithStats
        from vllm import SamplingParams
        
        logger.info("Successfully imported vLLM with stats logging")
        
        # Test with a small model for quick verification
        # You can replace this with your actual model path
        model_path = "facebook/opt-125m"  # Small test model
        
        logger.info(f"Initializing LLM with model: {model_path}")
        
        # Initialize LLM with stats logging
        # Use log_stats parameter (the standard way)
        llm = OfflineLLMWithStats(
            model=model_path,
            tensor_parallel_size=1,
            log_stats_interval=1,
            log_stats=True,  # Explicitly enable stats logging
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True  # Enable for prefix cache stats
        )
        
        logger.info("Testing with stats logging disabled parameter style...")
        # Alternative: Use disable_log_stats parameter (alternative way, for compatibility)
        # Uncomment to test with stats disabled
        # llm2 = OfflineLLMWithStats(
        #     model=model_path,
        #     tensor_parallel_size=1,
        #     log_stats_interval=1,
        #     disable_log_stats=True,  # Disable stats logging
        #     max_model_len=512,
        #     gpu_memory_utilization=0.8,
        #     enable_prefix_caching=True
        # )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        # Test prompts
        test_prompts = [
            "The capital of France is",
            "Machine learning is",
            "The weather today"
        ]
        
        logger.info("Starting inference test with stats logging...")
        
        # Run inference with detailed stats
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing prompt {i+1}: {prompt}")
            
            outputs = llm.generate([prompt], sampling_params, log_detailed_stats=True)
            
            # Print the result
            generated_text = outputs[0].outputs[0].text
            logger.info(f"Generated: {generated_text}")
        
        # Get final stats
        final_stats = llm.get_current_stats()
        logger.info(f"\n{'='*60}")
        logger.info("FINAL STATS SUMMARY:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("✅ vLLM stats logging test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vllm_stats_logging()
    sys.exit(0 if success else 1)
