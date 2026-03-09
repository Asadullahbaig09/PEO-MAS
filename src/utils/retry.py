"""
Retry mechanism for handling transient failures in API calls and scrapers
Implements exponential backoff with jitter
"""

import time
import logging
from typing import Callable, TypeVar, Any
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)"""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±20%)
            jitter_amount = delay * 0.2
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)


def retry_on_exception(
    config: RetryConfig = None,
    exceptions: tuple = (Exception,),
    log_level: str = 'warning'
) -> Callable:
    """
    Decorator for retrying a function on specific exceptions
    
    Usage:
        @retry_on_exception(
            config=RetryConfig(max_attempts=3),
            exceptions=(requests.RequestException, TimeoutError)
        )
        def fetch_data():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        log_msg = (
                            f"Attempt {attempt + 1}/{config.max_attempts} failed "
                            f"for {func.__name__}: {str(e)[:100]}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        if log_level == 'warning':
                            logger.warning(log_msg)
                        elif log_level == 'info':
                            logger.info(log_msg)
                        elif log_level == 'debug':
                            logger.debug(log_msg)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for "
                            f"{func.__name__}: {str(e)}"
                        )
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by failing fast after threshold is reached
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "CircuitBreaker"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = 'CLOSED'
        logger.debug(f"{self.name} circuit breaker reset to CLOSED")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(
                f"{self.name} circuit breaker opened after "
                f"{self.failure_count} failures"
            )
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == 'CLOSED':
            return True
        
        if self.state == 'OPEN':
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info(f"{self.name} circuit breaker moved to HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection
        
        Usage:
            breaker = CircuitBreaker(name="API_Call")
            result = breaker.execute(my_function, arg1, arg2)
        """
        if not self.can_execute():
            raise Exception(
                f"{self.name} circuit breaker is OPEN. "
                f"Service unavailable, will retry in {self.recovery_timeout}s"
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


# Prebuilt circuit breakers for common services
API_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    name="API"
)

SCRAPER_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    name="Scraper"
)

LLM_CIRCUIT_BREAKER = CircuitBreaker(
    failure_threshold=2,
    recovery_timeout=120,
    name="LLM"
)
