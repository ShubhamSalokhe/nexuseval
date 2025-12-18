import asyncio
import functools
import logging
from typing import Type, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    errors: Tuple[Type[Exception], ...] = (Exception,),
    on_error: Optional[Callable[[Exception, int], Any]] = None,
):
    """
    Decorator for async functions to retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries before giving up.
        initial_delay: Initial delay in seconds.
        backoff_factor: Multiplier for delay after each retry.
        errors: Tuple of exceptions to catch and retry on.
        on_error: Optional callback function invoked when an error occurs.
                  Signature: on_error(exception, attempt_number)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except errors as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    if on_error:
                        try:
                            on_error(e, attempt)
                        except Exception as cb_err:
                            logger.error(f"Error in retry callback: {cb_err}")

                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            
            # If we exhausted retries, re-raise the last exception
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator
