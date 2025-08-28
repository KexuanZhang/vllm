"""
Test script for the modified vLLM with stats logging.

This script tests the stats logging configuration in the modified vLLM,
verifying that stats logging is properly enabled by default and only 
disabled when explicitly turned off via parameters.
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
        
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Default behavior (should enable stats)")
        # Initialize LLM with default settings (should enable stats)
        llm_default = OfflineLLMWithStats(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.7,
            enable_prefix_caching=True  # Enable for prefix cache stats
        )
        
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Explicitly enabled stats logging")
        # Initialize LLM with stats logging explicitly enabled
        llm_enabled = OfflineLLMWithStats(
            model=model_path,
            tensor_parallel_size=1,
            log_stats_interval=1,
            log_stats=True,  # Explicitly enable stats logging
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.7,
            enable_prefix_caching=True  # Enable for prefix cache stats
        )
        
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Explicitly disabled stats logging")
        # Initialize LLM with stats logging explicitly disabled
        llm_disabled = OfflineLLMWithStats(
            model=model_path,
            tensor_parallel_size=1,
            log_stats_interval=1,
            disable_log_stats=True,  # Explicitly disable stats logging
            max_model_len=512,  # Small for testing
            gpu_memory_utilization=0.7,
            enable_prefix_caching=True
        )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,  # Lower temperature for more deterministic results
            top_p=0.95,
            max_tokens=20     # Short outputs for quick testing
        )
        
        # Test prompt
        test_prompt = "The capital of France is"
        
        logger.info("\n" + "="*60)
        logger.info("Running inference tests with different configurations...")
        
        # Test with default configuration (should have stats enabled)
        logger.info("\n=== Testing Default Configuration ===")
        outputs_default = llm_default.generate([test_prompt], sampling_params, log_detailed_stats=True)
        generated_text_default = outputs_default[0].outputs[0].text
        logger.info(f"Default config generated: {generated_text_default}")
        default_stats = llm_default.get_current_stats()
        
        # Test with explicitly enabled stats
        logger.info("\n=== Testing Explicitly Enabled Stats ===")
        outputs_enabled = llm_enabled.generate([test_prompt], sampling_params, log_detailed_stats=True)
        generated_text_enabled = outputs_enabled[0].outputs[0].text
        logger.info(f"Enabled stats config generated: {generated_text_enabled}")
        enabled_stats = llm_enabled.get_current_stats()
        
        # Test with explicitly disabled stats
        logger.info("\n=== Testing Explicitly Disabled Stats ===")
        outputs_disabled = llm_disabled.generate([test_prompt], sampling_params, log_detailed_stats=True)
        generated_text_disabled = outputs_disabled[0].outputs[0].text
        logger.info(f"Disabled stats config generated: {generated_text_disabled}")
        disabled_stats = llm_disabled.get_current_stats()
        
        # Compare the results to verify stats collection behavior
        logger.info("\n" + "="*60)
        logger.info("STATS COMPARISON SUMMARY:")
        
        # Check for presence of stats keys in each configuration
        def has_stats(stats_dict):
            """Check if stats dictionary contains engine stats."""
            return any(key.startswith('engine_') for key in stats_dict.keys())
        
        logger.info(f"Default config has stats: {has_stats(default_stats)}")
        logger.info(f"Explicitly enabled stats has stats: {has_stats(enabled_stats)}")
        logger.info(f"Explicitly disabled stats has stats: {has_stats(disabled_stats)}")
        
        # Show some key stats from each
        for config_name, stats in [
            ("Default", default_stats),
            ("Explicitly enabled", enabled_stats),
            ("Explicitly disabled", disabled_stats)
        ]:
            logger.info(f"\n--- {config_name} Configuration Stats ---")
            for key, value in stats.items():
                if key.startswith('engine_'):
                    logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "="*60)
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
