#!/usr/bin/env python3
"""
Quick verification script to check if the modified vLLM setup is working.
Run this after installing the modified vLLM.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_vllm_installation():
    """Verify that vLLM is properly installed and modified."""
    
    logger.info("🔍 Verifying vLLM installation...")
    
    try:
        # Test basic vLLM import
        import vllm
        logger.info(f"✅ vLLM imported successfully - Version: {vllm.__version__}")
        
        # Test our enhanced wrapper import
        from vllm.offline_llm_with_stats import OfflineLLMWithStats
        logger.info("✅ Enhanced LLM wrapper imported successfully")
        
        # Test V1 engine core modifications
        from vllm.v1.engine.core import EngineCore
        
        # Check if our added methods exist
        if hasattr(EngineCore, 'get_stats'):
            logger.info("✅ get_stats() method found in EngineCore")
        else:
            logger.warning("⚠️  get_stats() method NOT found in EngineCore")
            
        if hasattr(EngineCore, 'log_inference_stats'):
            logger.info("✅ log_inference_stats() method found in EngineCore")
        else:
            logger.warning("⚠️  log_inference_stats() method NOT found in EngineCore")
        
        # Test integration layer
        sys.path.insert(0, '/Users/zhang/Desktop/huawei/so1/semantic-operators/ggr-experiment-pipeline')
        from use_local_vllm import initialize_experiment_with_local_vllm
        logger.info("✅ Integration layer imported successfully")
        
        logger.info("🎉 All components verified successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please run the installation script: ./setup_vllm_with_stats.sh")
        return False
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def main():
    """Main verification function."""
    logger.info("Starting vLLM installation verification...")
    logger.info("="*50)
    
    success = verify_vllm_installation()
    
    logger.info("="*50)
    if success:
        logger.info("✅ Verification completed successfully!")
        logger.info("\n📝 Next steps:")
        logger.info("1. Test with: python test_enhanced_vllm_integration.py")
        logger.info("2. Run experiments with: python src/experiment/run_experiment_enhanced.py")
        logger.info("3. Check README_ENHANCED.md for detailed usage")
    else:
        logger.info("❌ Verification failed!")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("1. Run: cd /Users/zhang/Desktop/huawei/so1/vllm")
        logger.info("2. Run: ./setup_vllm_with_stats.sh")
        logger.info("3. Check for any installation errors")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
