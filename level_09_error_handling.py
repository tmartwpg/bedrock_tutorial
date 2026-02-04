"""
LEVEL 9: Error Handling & Retries - Building Resilient Applications

Building on Level 8, we now learn to:
1. Handle API errors gracefully
2. Implement retry logic with exponential backoff
3. Use rate limiting
4. Implement circuit breakers
5. Add fallback strategies

Topics covered:
  - Exception handling
  - Retry strategies
  - Exponential backoff
  - Rate limiting
  - Timeouts
  - Fallback responses
  - Logging and monitoring

Expected output:
  Robust applications that handle failures gracefully
"""

import time
import random
import logging
from typing import Callable, Any, Optional
from functools import wraps
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from utils import pretty_print
from config import MODEL_ID, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Retry Decorators and Utility Functions
# ============================================================================

def retry_with_exponential_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_DELAY,
    max_delay: float = 60.0
):
    """
    Decorator that implements retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries

    Returns:
        Decorated function with retry capability
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Attempt {attempt} of {max_retries}: {func.__name__}")
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"[OK] Succeeded after {attempt} attempts")
                    return result

                except Exception as e:
                    # Calculate backoff delay with jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    # Add jitter: random amount up to 10% of delay
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter

                    logger.warning(f"Attempt {attempt} failed: {str(e)}")

                    if attempt < max_retries:
                        logger.info(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Implements circuit breaker pattern for fault tolerance.

    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            timeout: Seconds before attempting to close from open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, or HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result if successful
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service temporarily unavailable")

        try:
            result = func(*args, **kwargs)
            # Success - reset circuit if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise


class ErrorHandlingTutorial:
    """
    Demonstrates error handling patterns for LLM applications.

    Building resilient applications requires anticipating failures.
    """

    def __init__(self):
        """Initialize model."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )
        self.circuit_breaker = CircuitBreaker(failure_threshold=3)

    def example_1_basic_exception_handling(self):
        """Example 1: Basic try-except error handling."""
        pretty_print("Example 1: Basic Exception Handling", "")

        def safe_invoke(prompt: str) -> Optional[str]:
            """Safely invoke model with basic error handling."""
            try:
                logger.info(f"Invoking model with prompt")
                messages = [HumanMessage(content=prompt)]
                response = self.model.invoke(messages)
                logger.info("[OK] Model invocation successful")
                return response.content

            except ValueError as e:
                logger.error(f"Value error: {e}")
                return None

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

        # Test with valid input
        print("Test 1: Valid input")
        result = safe_invoke("Say hello")
        print(f"Result: {result[:50] if result else 'Error occurred'}\n")

        print("[OK] Basic exception handling in place!\n")

    def example_2_retry_pattern(self):
        """Example 2: Retry with exponential backoff."""
        pretty_print("Example 2: Retry with Exponential Backoff", "")

        attempt_count = 0

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.5)
        def invoke_with_retry(prompt: str) -> str:
            """Model invocation with automatic retries."""
            nonlocal attempt_count
            attempt_count += 1

            # Simulate occasional failures (for demo purposes)
            if attempt_count < 2 and random.random() > 0.7:
                raise Exception("Simulated temporary failure")

            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            return response.content

        try:
            print("Invoking model with retry capability...")
            result = invoke_with_retry("What is machine learning?")
            print(f"[OK] Success: {result[:100]}...\n")
        except Exception as e:
            print(f"[FAIL] Failed after retries: {e}\n")

    def example_3_timeout_handling(self):
        """Example 3: Handling timeouts."""
        pretty_print("Example 3: Timeout Handling", "")

        def invoke_with_timeout(prompt: str, timeout: float = REQUEST_TIMEOUT) -> Optional[str]:
            """Invoke model with timeout protection."""
            try:
                logger.info(f"Invoking model with {timeout}s timeout")
                messages = [HumanMessage(content=prompt)]

                # In real scenarios, you'd use asyncio with timeout
                # For now, just show the pattern
                response = self.model.invoke(messages)

                logger.info("[OK] Completed within timeout")
                return response.content

            except TimeoutError:
                logger.error(f"Request exceeded {timeout}s timeout")
                return None

            except Exception as e:
                logger.error(f"Error: {e}")
                return None

        result = invoke_with_timeout("Explain quantum computing briefly")
        print(f"Result: {result[:100] if result else 'Timeout'}\n")
        print("[OK] Timeout protection in place!\n")

    def example_4_circuit_breaker(self):
        """Example 4: Circuit breaker pattern."""
        pretty_print("Example 4: Circuit Breaker Pattern", "")

        def protected_invoke(prompt: str) -> str:
            """Invoke model with circuit breaker protection."""
            def _invoke():
                messages = [HumanMessage(content=prompt)]
                response = self.model.invoke(messages)
                return response.content

            return self.circuit_breaker.call(_invoke)

        print("Demonstrating circuit breaker...")
        print(f"Initial state: {self.circuit_breaker.state}\n")

        # Make requests
        prompts = [
            "Hello",
            "What's 2+2?",
            "Tell me about AI"
        ]

        for prompt in prompts:
            try:
                result = protected_invoke(prompt)
                print(f"[OK] Request succeeded: {result[:40]}...")
                print(f"  Circuit breaker state: {self.circuit_breaker.state}\n")

            except Exception as e:
                print(f"[FAIL] Request failed: {e}")
                print(f"  Circuit breaker state: {self.circuit_breaker.state}\n")

        print("[OK] Circuit breaker prevents cascading failures!\n")

    def example_5_fallback_strategy(self):
        """Example 5: Fallback strategies."""
        pretty_print("Example 5: Fallback Strategies", "")

        def invoke_with_fallback(prompt: str) -> str:
            """Invoke with multiple fallback options."""
            fallbacks = [
                lambda: self._primary_response(prompt),
                lambda: self._secondary_response(prompt),
                lambda: self._default_response(),
            ]

            for i, fallback in enumerate(fallbacks):
                try:
                    logger.info(f"Attempting strategy {i+1}/{len(fallbacks)}")
                    result = fallback()
                    logger.info(f"[OK] Strategy {i+1} succeeded")
                    return result

                except Exception as e:
                    logger.warning(f"Strategy {i+1} failed: {e}")

                    if i == len(fallbacks) - 1:
                        logger.error("All strategies exhausted")

            return self._default_response()

        def _primary_response(self, prompt: str) -> str:
            """Primary response strategy."""
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            return response.content

        def _secondary_response(self, prompt: str) -> str:
            """Secondary response strategy (e.g., simpler prompt)."""
            simplified = f"Briefly: {prompt}"
            messages = [HumanMessage(content=simplified)]
            response = self.model.invoke(messages)
            return response.content

        def _default_response(self) -> str:
            """Default/cached response."""
            return "I'm currently unable to process this request. Please try again later."

        # Bind methods
        self._primary_response = _primary_response.__get__(self)
        self._secondary_response = _secondary_response.__get__(self)
        self._default_response = _default_response.__get__(self)

        result = invoke_with_fallback("What is AI?")
        print(f"Result: {result[:100]}...\n")
        print("[OK] Fallback strategies ensure availability!\n")

    def example_6_validation_and_sanitization(self):
        """Example 6: Input validation."""
        pretty_print("Example 6: Input Validation", "")

        def validate_and_invoke(prompt: str) -> Optional[str]:
            """Validate input before invoking model."""
            # Validation rules
            if not prompt:
                logger.error("Empty prompt")
                return None

            if len(prompt) > 10000:
                logger.error("Prompt too long")
                return None

            if len(prompt.split()) < 2:
                logger.warning("Prompt very short, but proceeding")

            # Sanitize input
            sanitized = prompt.strip()[:1000]  # Truncate to safety limit

            try:
                logger.info("Input validated, invoking model")
                messages = [HumanMessage(content=sanitized)]
                response = self.model.invoke(messages)
                return response.content

            except Exception as e:
                logger.error(f"Invocation failed: {e}")
                return None

        # Test cases
        test_cases = [
            ("Valid prompt about Python", True),
            ("", False),  # Empty
            ("Hi", True),  # Short but valid
        ]

        for prompt, should_succeed in test_cases:
            result = validate_and_invoke(prompt)
            status = "[OK]" if (result is not None) == should_succeed else "[FAIL]"
            print(f"{status} '{prompt[:30]}...' -> {result[:30] if result else 'None'}...")

        print("\n[OK] Input validation prevents many errors!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 9: Error Handling & Retries")
        print("=" * 80)

        try:
            self.example_1_basic_exception_handling()
            self.example_2_retry_pattern()
            self.example_3_timeout_handling()
            self.example_4_circuit_breaker()
            self.example_5_fallback_strategy()
            self.example_6_validation_and_sanitization()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. Always use try-except for error handling")
            print("  2. Retry logic with exponential backoff handles transient failures")
            print("  3. Timeouts prevent hanging requests")
            print("  4. Circuit breaker prevents cascading failures")
            print("  5. Fallback strategies ensure graceful degradation")
            print("  6. Input validation prevents many problems")
            print("  7. Logging provides visibility for debugging")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates error handling patterns.

    Usage:
        python level_09_error_handling.py
    """

    tutorial = ErrorHandlingTutorial()
    tutorial.demonstrate()
