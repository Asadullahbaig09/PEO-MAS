"""
Unit tests for retry and error handling logic
"""

import pytest
import time
from unittest.mock import Mock, patch, call

from src.utils.retry import (
    RetryConfig,
    retry_on_exception,
    CircuitBreaker,
    API_CIRCUIT_BREAKER
)


class TestRetryConfig:
    """Test retry configuration"""
    
    def test_retry_config_creation(self):
        """Test retry config initializes correctly"""
        config = RetryConfig(max_attempts=5, initial_delay=0.5)
        
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
    
    def test_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            max_attempts=4,
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # First retry: 1 * 2^0 = 1
        assert config.get_delay(0) == 1.0
        
        # Second retry: 1 * 2^1 = 2
        assert config.get_delay(1) == 2.0
        
        # Third retry: 1 * 2^2 = 4
        assert config.get_delay(2) == 4.0
    
    def test_max_delay_enforcement(self):
        """Test maximum delay is enforced"""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=10.0,
            jitter=False
        )
        
        # High attempt number would exceed max_delay
        delay = config.get_delay(10)
        assert delay <= config.max_delay
    
    def test_jitter_adds_randomness(self):
        """Test jitter adds variance to delays"""
        config = RetryConfig(
            initial_delay=100.0,
            jitter=True
        )
        
        delays = [config.get_delay(0) for _ in range(10)]
        
        # With jitter, delays should vary
        assert len(set(delays)) > 1  # Not all same value


class TestRetryDecorator:
    """Test retry decorator functionality"""
    
    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt"""
        config = RetryConfig(max_attempts=3)
        
        @retry_on_exception(config=config)
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    def test_retry_on_exception(self):
        """Test function is retried on exception"""
        config = RetryConfig(max_attempts=3, initial_delay=0.01)
        call_count = 0
        
        @retry_on_exception(config=config)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = failing_func()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test exception raised after max retries"""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        @retry_on_exception(config=config)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
    
    def test_specific_exception_types(self):
        """Test retrying only specific exception types"""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        @retry_on_exception(
            config=config,
            exceptions=(ValueError,)
        )
        def sometimes_fails():
            raise TypeError("This won't be retried")
        
        # TypeError not in exceptions tuple, should raise immediately
        with pytest.raises(TypeError):
            sometimes_fails()
    
    def test_retry_delay_between_attempts(self):
        """Test delay occurs between retry attempts"""
        config = RetryConfig(max_attempts=3, initial_delay=0.05)
        
        attempt_times = []
        
        @retry_on_exception(config=config)
        def func_with_delays():
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise ValueError()
            return "done"
        
        func_with_delays()
        
        # Should have 3 attempts
        assert len(attempt_times) == 3
        
        # Time between attempts should be > delay
        time_between_1_2 = attempt_times[1] - attempt_times[0]
        time_between_2_3 = attempt_times[2] - attempt_times[1]
        
        assert time_between_1_2 >= 0.04  # Allow small margin
        assert time_between_2_3 >= 0.08  # Exponential backoff


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker initializes correctly"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker allows execution in CLOSED state"""
        breaker = CircuitBreaker()
        
        assert breaker.can_execute() is True
        assert breaker.state == 'CLOSED'
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == 'OPEN'
        assert breaker.can_execute() is False
    
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets after success"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Open the circuit
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == 'OPEN'
        
        # Let timeout pass and record success
        breaker.last_failure_time = time.time() - 100  # Timeout passed
        assert breaker.can_execute() is True  # Half-open state
        
        breaker.record_success()
        assert breaker.state == 'CLOSED'
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_prevents_cascade(self):
        """Test circuit breaker prevents cascading failures"""
        breaker = CircuitBreaker(failure_threshold=2)
        
        # Fail twice to open circuit
        breaker.record_failure()
        breaker.record_failure()
        
        # Circuit is now open
        assert breaker.can_execute() is False
        
        # Function calls should fail fast without executing
        def failing_function():
            raise Exception("Should not reach here")
        
        with pytest.raises(Exception):
            breaker.execute(failing_function)
    
    def test_circuit_breaker_execute_method(self):
        """Test execute method with circuit breaker"""
        breaker = CircuitBreaker()
        
        def add(a, b):
            return a + b
        
        result = breaker.execute(add, 2, 3)
        assert result == 5
    
    def test_circuit_breaker_with_failing_function(self):
        """Test execute with function that fails"""
        breaker = CircuitBreaker(failure_threshold=1)
        
        def failing_func():
            raise ValueError("Function failed")
        
        # First call fails and opens circuit
        with pytest.raises(ValueError):
            breaker.execute(failing_func)
        
        assert breaker.state == 'OPEN'
        
        # Second call should fail with circuit breaker error
        with pytest.raises(Exception):
            breaker.execute(failing_func)


class TestPredefinedCircuitBreakers:
    """Test predefined circuit breakers"""
    
    def test_api_circuit_breaker_exists(self):
        """Test API circuit breaker is available"""
        assert API_CIRCUIT_BREAKER is not None
        assert API_CIRCUIT_BREAKER.name == "API"
    
    def test_prebuilt_breakers_have_correct_settings(self):
        """Test prebuilt breakers have reasonable settings"""
        # API breaker should be strict (fail fast on 3 errors)
        assert API_CIRCUIT_BREAKER.failure_threshold == 3
        assert API_CIRCUIT_BREAKER.recovery_timeout > 0


class TestErrorHandlingIntegration:
    """Test error handling in realistic scenarios"""
    
    def test_retry_then_circuit_breaker(self):
        """Test combining retry decorator with circuit breaker"""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        breaker = CircuitBreaker(failure_threshold=2)
        
        call_count = 0
        
        @retry_on_exception(config=config, exceptions=(ValueError,))
        def api_call():
            nonlocal call_count
            call_count += 1
            raise ValueError("API error")
        
        # First two calls should retry
        with pytest.raises(ValueError):
            api_call()
        
        # Record circuit breaker failure
        breaker.record_failure()
        
        with pytest.raises(ValueError):
            api_call()
        
        breaker.record_failure()
        
        # Now circuit is open
        assert breaker.state == 'OPEN'
        
        # Further calls fail fast
        with pytest.raises(Exception):
            breaker.execute(api_call)
    
    def test_graceful_degradation_pattern(self):
        """Test graceful degradation with retry and fallback"""
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        
        @retry_on_exception(config=config)
        def primary_source():
            raise ConnectionError("Primary source unavailable")
        
        def fallback_source():
            return "fallback data"
        
        try:
            data = primary_source()
        except ConnectionError:
            data = fallback_source()
        
        assert data == "fallback data"
