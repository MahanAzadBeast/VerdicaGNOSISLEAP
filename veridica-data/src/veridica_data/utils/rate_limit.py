"""
API Rate Limiting Utilities
Implements QPS (queries per second) limiting for API calls
"""

import time
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def qps_limiter(max_qps: float) -> Callable:
    """
    Decorator to limit function calls to max_qps queries per second
    
    Args:
        max_qps: Maximum queries per second allowed
        
    Returns:
        Decorated function with rate limiting
    """
    interval = 1.0 / max_qps
    last = [0.0]
    
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs) -> Any:
            now = time.time()
            wait = interval - (now - last[0])
            if wait > 0:
                logger.debug(f"Rate limiting: waiting {wait:.2f}s")
                time.sleep(wait)
            last[0] = time.time()
            return fn(*args, **kwargs)
        return wrapped
    return decorator


class RateLimiter:
    """
    Context manager for rate limiting API calls
    """
    
    def __init__(self, max_qps: float):
        self.interval = 1.0 / max_qps
        self.last_call = 0.0
    
    def __enter__(self):
        now = time.time()
        wait = self.interval - (now - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def adaptive_rate_limiter(base_qps: float, backoff_factor: float = 0.5):
    """
    Adaptive rate limiter that backs off on errors
    
    Args:
        base_qps: Base queries per second
        backoff_factor: Factor to reduce QPS on errors
        
    Returns:
        Rate limiter that adapts to API responses
    """
    current_qps = [base_qps]
    error_count = [0]
    
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs) -> Any:
            # Apply current rate limit
            interval = 1.0 / current_qps[0]
            time.sleep(interval)
            
            try:
                result = fn(*args, **kwargs)
                # Success - gradually increase QPS back to base
                if current_qps[0] < base_qps:
                    current_qps[0] = min(base_qps, current_qps[0] * 1.1)
                error_count[0] = 0
                return result
                
            except Exception as e:
                # Error - back off
                error_count[0] += 1
                if error_count[0] > 3:
                    current_qps[0] *= backoff_factor
                    logger.warning(f"Backing off to {current_qps[0]:.2f} QPS due to errors")
                raise
                
        return wrapped
    return decorator