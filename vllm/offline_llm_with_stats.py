"""
Enhanced offline LLM with detailed stats logging for experimentation.
"""
import logging
import time
from typing import List, Optional, Union, Dict, Any
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)

class OfflineLLMWithStats:
    """Wrapper around vLLM's LLM class that exposes detailed inference statistics."""
    
    def __init__(self, 
                 model: str,
                 tensor_parallel_size: int = 1,
                 log_stats_interval: int = 1,  # Log stats every N requests
                 log_stats: bool = True,  # Flag to enable stats logging (not passed to LLM)
                 **kwargs):
        """
        Initialize the LLM with stats logging enabled.
        
        Args:
            model: Model path or name
            tensor_parallel_size: Number of GPUs for tensor parallelism
            log_stats_interval: How often to log stats (every N requests)
            log_stats: Whether to log stats (not passed to LLM)
            **kwargs: Additional arguments passed to LLM constructor
        """
        # Store log_stats setting but don't pass it to LLM
        self._log_stats = log_stats
        
        # Filter out our custom parameters that aren't accepted by EngineArgs
        llm_kwargs = kwargs.copy()
        if 'log_stats' in llm_kwargs:
            llm_kwargs.pop('log_stats')
        
        self.llm = LLM(model=model, 
                      tensor_parallel_size=tensor_parallel_size, 
                      **llm_kwargs)
        self.log_stats_interval = log_stats_interval
        self._request_counter = 0
        self._total_inference_time = 0.0
        
    def generate(self, 
                prompts: Union[str, List[str]], 
                sampling_params: Optional[SamplingParams] = None,
                log_detailed_stats: bool = True) -> List[RequestOutput]:
        """
        Generate responses with detailed stats logging.
        
        Args:
            prompts: Input prompts
            sampling_params: Sampling parameters
            log_detailed_stats: Whether to log detailed stats for this generation
            
        Returns:
            List of RequestOutput objects
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        start_time = time.time()
        
        # Only log stats if the master log_stats setting is enabled
        should_log = getattr(self, '_log_stats', True) and log_detailed_stats
        
        # Log pre-generation stats
        if should_log:
            logger.info(f"=== Pre-Generation Stats (Batch size: {len(prompts)}) ===")
            self._log_engine_stats("PRE-GEN")
        
        # Run inference
        outputs = self.llm.generate(prompts, sampling_params)
        
        end_time = time.time()
        inference_time = end_time - start_time
        self._total_inference_time += inference_time
        self._request_counter += len(prompts)
        
        # Log post-generation stats if logging is enabled
        should_log = getattr(self, '_log_stats', True) and (
            log_detailed_stats or (self._request_counter % self.log_stats_interval == 0)
        )
        
        if should_log:
            logger.info(f"=== Post-Generation Stats ===")
            self._log_engine_stats("POST-GEN")
            self._log_performance_stats(len(prompts), inference_time)
            
        return outputs
    
    def _log_engine_stats(self, prefix: str = ""):
        """Log detailed engine statistics."""
        try:
            # Access the engine core to get stats
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                
                # Check if we're using V1 engine with EngineCore
                if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'get_stats'):
                    # V1 engine with EngineCore stats
                    stats = engine.engine_core.get_stats()
                    if stats:
                        self._format_and_log_stats_v1(stats, prefix)
                        return
                elif hasattr(engine, 'get_stats'):
                    # V0 engine or direct stats method
                    stats = engine.get_stats()
                    if stats:
                        self._format_and_log_stats_v0(stats, prefix)
                        return
                elif hasattr(engine, '_get_stats'):
                    # V0 legacy _get_stats method
                    stats = engine._get_stats()
                    if stats:
                        self._format_and_log_stats_v0(stats, prefix)
                        return
                
                # Try to access scheduler stats directly
                if hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'make_stats'):
                    stats = engine.scheduler.make_stats()
                    if stats:
                        self._format_and_log_stats_v1(stats, prefix)
                        return
                        
                logger.warning(f"{prefix}: Unable to access engine stats - no known stats method found")
            else:
                logger.warning(f"{prefix}: Unable to access llm_engine")
        except Exception as e:
            logger.error(f"{prefix}: Error accessing engine stats: {e}")
            import traceback
            traceback.print_exc()
    
    def _format_and_log_stats_v1(self, stats: Any, prefix: str = ""):
        """Format and log V1 stats object (SchedulerStats)."""
        logger.info(f"{prefix} V1 Engine Statistics:")
        
        # KV Cache Usage
        if hasattr(stats, 'kv_cache_usage'):
            cache_usage = stats.kv_cache_usage * 100
            logger.info(f"  GPU KV Cache Usage: {cache_usage:.2f}%")
        
        # Prefix Cache Hit Rate
        if hasattr(stats, 'prefix_cache_stats') and stats.prefix_cache_stats:
            prefix_stats = stats.prefix_cache_stats
            if prefix_stats.queries > 0:
                hit_rate = (prefix_stats.hits / prefix_stats.queries) * 100
                logger.info(f"  Prefix Cache Hit Rate: {hit_rate:.2f}% ({prefix_stats.hits}/{prefix_stats.queries})")
            else:
                logger.info(f"  Prefix Cache: No queries yet")
        
        # Request Queue Stats
        if hasattr(stats, 'num_running_reqs'):
            logger.info(f"  Running Requests: {stats.num_running_reqs}")
        if hasattr(stats, 'num_waiting_reqs'):
            logger.info(f"  Waiting Requests: {stats.num_waiting_reqs}")
            
        # Additional metrics if available
        for attr in ['num_corrupted_reqs', 'step_counter', 'current_wave']:
            if hasattr(stats, attr):
                logger.info(f"  {attr}: {getattr(stats, attr)}")
                
        # Speculative decoding stats
        if hasattr(stats, 'spec_decoding_stats') and stats.spec_decoding_stats:
            spec_stats = stats.spec_decoding_stats
            logger.info(f"  Speculative Decoding Stats: {spec_stats}")
    
    def _format_and_log_stats_v0(self, stats: Any, prefix: str = ""):
        """Format and log V0 stats object (Legacy Stats)."""
        logger.info(f"{prefix} V0 Engine Statistics:")
        
        # V0 stats structure (different attribute names)
        if hasattr(stats, 'gpu_cache_usage_sys'):
            cache_usage = stats.gpu_cache_usage_sys * 100
            logger.info(f"  GPU KV Cache Usage: {cache_usage:.2f}%")
        
        if hasattr(stats, 'gpu_prefix_cache_hit_rate'):
            hit_rate = stats.gpu_prefix_cache_hit_rate * 100
            logger.info(f"  GPU Prefix Cache Hit Rate: {hit_rate:.2f}%")
            
        if hasattr(stats, 'cpu_prefix_cache_hit_rate'):
            hit_rate = stats.cpu_prefix_cache_hit_rate * 100
            logger.info(f"  CPU Prefix Cache Hit Rate: {hit_rate:.2f}%")
        
        # Request Queue Stats
        if hasattr(stats, 'num_running_sys'):
            logger.info(f"  Running Requests: {stats.num_running_sys}")
        if hasattr(stats, 'num_waiting_sys'):
            logger.info(f"  Waiting Requests: {stats.num_waiting_sys}")
        if hasattr(stats, 'num_swapped_sys'):
            logger.info(f"  Swapped Requests: {stats.num_swapped_sys}")
            
        # Additional V0 metrics
        for attr in ['cpu_cache_usage_sys', 'num_preemptions', 'num_prompt_tokens_iter', 'num_generation_tokens_iter']:
            if hasattr(stats, attr):
                logger.info(f"  {attr}: {getattr(stats, attr)}")

    def _format_and_log_stats(self, stats: Any, prefix: str = ""):
        """Format and log stats object - auto-detect V0 vs V1."""
        # Auto-detect stats format based on available attributes
        if hasattr(stats, 'kv_cache_usage') or hasattr(stats, 'prefix_cache_stats'):
            # V1 format
            self._format_and_log_stats_v1(stats, prefix)
        elif hasattr(stats, 'gpu_cache_usage_sys') or hasattr(stats, 'num_running_sys'):
            # V0 format
            self._format_and_log_stats_v0(stats, prefix)
        else:
            logger.info(f"{prefix} Unknown stats format: {stats}")
            logger.info(f"  Available attributes: {[attr for attr in dir(stats) if not attr.startswith('_')]}")
    
    def _log_performance_stats(self, batch_size: int, inference_time: float):
        """Log performance statistics."""
        throughput = batch_size / inference_time if inference_time > 0 else 0
        avg_time_per_request = self._total_inference_time / self._request_counter if self._request_counter > 0 else 0
        
        logger.info(f"Performance Statistics:")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Inference Time: {inference_time:.3f}s")
        logger.info(f"  Throughput: {throughput:.2f} requests/sec")
        logger.info(f"  Total Requests Processed: {self._request_counter}")
        logger.info(f"  Average Time per Request: {avg_time_per_request:.3f}s")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current engine statistics as a dictionary."""
        stats_dict = {
            'total_requests': self._request_counter,
            'total_inference_time': self._total_inference_time,
            'avg_time_per_request': self._total_inference_time / max(1, self._request_counter)
        }
        
        # Only attempt to get detailed engine stats if logging is enabled
        if not getattr(self, '_log_stats', True):
            return stats_dict
            
        try:
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                engine_stats = None
                
                # Try different methods to get stats
                if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'get_stats'):
                    # V1 engine with EngineCore
                    engine_stats = engine.engine_core.get_stats()
                elif hasattr(engine, 'get_stats'):
                    # Direct get_stats method
                    engine_stats = engine.get_stats()
                elif hasattr(engine, '_get_stats'):
                    # Legacy _get_stats method
                    engine_stats = engine._get_stats()
                elif hasattr(engine, 'scheduler') and hasattr(engine.scheduler, 'make_stats'):
                    # Direct scheduler access
                    engine_stats = engine.scheduler.make_stats()
                
                if engine_stats:
                    # Convert stats object to dict
                    self._extract_stats_to_dict(engine_stats, stats_dict)
                else:
                    logger.warning("No engine stats available")
                    
        except Exception as e:
            logger.error(f"Error collecting stats: {e}")
            
        return stats_dict
    
    def _extract_stats_to_dict(self, stats: Any, stats_dict: Dict[str, Any]):
        """Extract stats object attributes into dictionary."""
        for attr in dir(stats):
            if attr.startswith('_') or callable(getattr(stats, attr)):
                continue
                
            try:
                value = getattr(stats, attr)
                
                # Handle simple types
                if isinstance(value, (int, float, str, bool)):
                    stats_dict[f'engine_{attr}'] = value
                # Handle nested objects (like PrefixCacheStats)
                elif hasattr(value, '__dict__') and not callable(value):
                    for nested_attr in dir(value):
                        if not nested_attr.startswith('_') and not callable(getattr(value, nested_attr)):
                            try:
                                nested_value = getattr(value, nested_attr)
                                if isinstance(nested_value, (int, float, str, bool)):
                                    stats_dict[f'engine_{attr}_{nested_attr}'] = nested_value
                            except:
                                pass
                # Handle lists/tuples of simple types
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if all(isinstance(x, (int, float, str, bool)) for x in value):
                        stats_dict[f'engine_{attr}'] = value
            except:
                pass
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying LLM."""
        return getattr(self.llm, name)
