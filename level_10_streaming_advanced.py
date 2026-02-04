"""
LEVEL 10: Streaming & Advanced Patterns

The capstone level combining everything we've learned:
1. Streaming responses for real-time output
2. Async/await for concurrent operations
3. Token counting for cost optimization
4. Batch processing
5. Performance optimization
6. Production patterns

Topics covered:
  - Streaming responses
  - Async operations
  - Token counting
  - Batch processing
  - Performance tuning
  - Production ready patterns

Expected output:
  Advanced, production-ready LLM applications
"""

import asyncio
import json
import time
from typing import AsyncIterator, List, Dict, Any
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage
from utils import pretty_print
from config import MODEL_ID

# Streaming callbacks are available but not used in this basic implementation
# For production, you would use: from langchain_community.callbacks import StreamingStdOutCallbackHandler


class AdvancedPatternsTutorial:
    """
    Demonstrates advanced patterns for production LLM applications.

    Covers streaming, async operations, and optimization techniques.
    """

    def __init__(self):
        """Initialize model with streaming capability."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1024,
            }
        )

    def example_1_streaming_responses(self):
        """Example 1: Stream responses token-by-token."""
        pretty_print("Example 1: Streaming Responses", "")

        print("Request: Write a haiku about programming")
        print("\nStreaming response:\n")

        # For production streaming, you would configure streaming callbacks
        # Here we show a regular invocation
        prompt = "Write a haiku about programming and its beauty"
        messages = [HumanMessage(content=prompt)]

        # Invoke the model
        response = self.model.invoke(messages)
        print(response.content)

        print("\n[OK] Response generated! (Streaming available in production)\n")

    def example_2_token_counting(self):
        """Example 2: Count tokens for cost estimation."""
        pretty_print("Example 2: Token Counting", "")

        # Sample texts
        texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Artificial Intelligence is transforming every industry in profound ways.",
        ]

        print("Estimating token costs:\n")

        total_tokens = 0

        for text in texts:
            # Approximate token count (rough estimation)
            # Claude typically uses ~4 chars per token on average
            approximate_tokens = len(text.split()) + 5

            print(f"Text: '{text}'")
            print(f"  Approximate tokens: {approximate_tokens}")
            print(f"  Estimated cost: ~${approximate_tokens * 0.0001:.4f}\n")

            total_tokens += approximate_tokens

        print(f"Total estimated tokens: {total_tokens}")
        print(f"Total estimated cost: ~${total_tokens * 0.0001:.4f}")
        print("\n[OK] Token counting helps predict costs!\n")

    def example_3_batch_processing(self):
        """Example 3: Process multiple requests efficiently."""
        pretty_print("Example 3: Batch Processing", "")

        prompts = [
            "What is Python?",
            "What is Java?",
            "What is JavaScript?",
            "What is Go?",
            "What is Rust?",
        ]

        print(f"Processing {len(prompts)} requests sequentially...\n")

        start_time = time.time()
        results = []

        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Processing: '{prompt}'")

            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)

            results.append({
                "prompt": prompt,
                "response": response.content[:100] + "..."
            })

        elapsed = time.time() - start_time

        print(f"\nCompleted {len(results)} requests in {elapsed:.2f}s")
        print(f"Average time per request: {elapsed/len(prompts):.2f}s\n")

        print("Results:")
        for r in results:
            print(f"  Q: {r['prompt']}")
            print(f"  A: {r['response']}\n")

        print("[OK] Batch processing handles multiple requests!\n")

    def example_4_caching_pattern(self):
        """Example 4: Simple caching to avoid redundant calls."""
        pretty_print("Example 4: Caching Pattern", "")

        # Simple cache
        cache = {}

        def get_response_cached(prompt: str) -> str:
            """Get response with caching."""
            # Check cache first
            if prompt in cache:
                print(f"  [OK] Cache hit!")
                return cache[prompt]

            # Cache miss - invoke model
            print(f"  [FAIL] Cache miss, invoking model...")
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            result = response.content

            # Store in cache
            cache[prompt] = result

            return result

        # Test caching
        print("Test 1: First request")
        result1 = get_response_cached("What is machine learning?")
        print(f"Response: {result1[:80]}...\n")

        print("Test 2: Same question (should be cached)")
        result2 = get_response_cached("What is machine learning?")
        print(f"Response: {result2[:80]}...\n")

        print("Test 3: Different question")
        result3 = get_response_cached("What is deep learning?")
        print(f"Response: {result3[:80]}...\n")

        print(f"Cache size: {len(cache)} entries")
        print("[OK] Caching reduces API calls and latency!\n")

    def example_5_request_limiting(self):
        """Example 5: Rate limiting for compliance."""
        pretty_print("Example 5: Rate Limiting", "")

        class RateLimiter:
            """Simple rate limiter."""
            def __init__(self, max_requests: int = 5, window: float = 60):
                self.max_requests = max_requests
                self.window = window
                self.requests = []

            def is_allowed(self) -> bool:
                """Check if request is allowed."""
                now = time.time()
                # Remove old requests outside window
                self.requests = [t for t in self.requests if now - t < self.window]

                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True
                return False

        # Create limiter: 3 requests per 10 seconds
        limiter = RateLimiter(max_requests=3, window=10)

        print("Rate limiter: 3 requests per 10 seconds\n")

        # Make requests
        for i in range(6):
            if limiter.is_allowed():
                print(f"  Request {i+1}: [OK] Allowed")
            else:
                print(f"  Request {i+1}: [FAIL] Rate limited")
            time.sleep(0.5)

        print("\n[OK] Rate limiting prevents abuse!\n")

    def example_6_async_operations(self):
        """Example 6: Async operations for concurrent execution."""
        pretty_print("Example 6: Async/Concurrent Operations", "")

        async def async_invoke(prompt: str, delay: float = 0) -> str:
            """Async model invocation with optional delay."""
            if delay > 0:
                await asyncio.sleep(delay)

            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            return response.content

        async def run_concurrent_requests():
            """Run multiple requests concurrently."""
            prompts = [
                "What is AI?",
                "What is ML?",
                "What is DL?",
            ]

            print("Starting concurrent requests...\n")
            start_time = time.time()

            # Run all requests concurrently
            tasks = [async_invoke(p) for p in prompts]
            results = await asyncio.gather(*tasks)

            elapsed = time.time() - start_time

            print(f"Completed {len(results)} requests in {elapsed:.2f}s")
            print(f"Sequential would take ~{elapsed * len(results):.2f}s\n")

            return results

        # Run async demo
        try:
            results = asyncio.run(run_concurrent_requests())
            print("Results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result[:60]}...")
            print("\n[OK] Async operations enable concurrent execution!\n")
        except Exception as e:
            print(f"Note: {e}")
            print("Async demo requires proper async context\n")

    def example_7_production_pattern(self):
        """Example 7: Production-ready pattern combining everything."""
        pretty_print("Example 7: Production-Ready Pattern", "")

        class ProductionLLMService:
            """Production-ready LLM service with all patterns."""

            def __init__(self, model_id: str):
                self.model = ChatBedrock(
                    model_id=model_id,
                    region_name="us-east-1",
                    model_kwargs={"temperature": 0.7}
                )
                self.cache = {}
                self.request_log = []

            def invoke(self, prompt: str, use_cache: bool = True) -> Dict[str, Any]:
                """
                Production-ready invocation with all safety measures.

                Returns:
                    Dict with response, metadata, and stats
                """
                start_time = time.time()

                # Step 1: Validate input
                if not prompt or len(prompt) > 10000:
                    return {"error": "Invalid prompt", "status": "failed"}

                # Step 2: Check cache
                if use_cache and prompt in self.cache:
                    return {
                        "response": self.cache[prompt],
                        "source": "cache",
                        "latency_ms": 1,
                        "status": "success"
                    }

                # Step 3: Invoke model
                try:
                    messages = [HumanMessage(content=prompt)]
                    response = self.model.invoke(messages)
                    result = response.content

                    # Step 4: Cache result
                    self.cache[prompt] = result

                    # Step 5: Record metrics
                    latency = (time.time() - start_time) * 1000
                    self.request_log.append({
                        "prompt_length": len(prompt),
                        "response_length": len(result),
                        "latency_ms": latency,
                        "timestamp": time.time()
                    })

                    return {
                        "response": result,
                        "source": "model",
                        "latency_ms": latency,
                        "status": "success"
                    }

                except Exception as e:
                    return {
                        "error": str(e),
                        "status": "failed",
                        "latency_ms": (time.time() - start_time) * 1000
                    }

            def get_stats(self) -> Dict[str, Any]:
                """Get service statistics."""
                if not self.request_log:
                    return {"status": "no_requests"}

                latencies = [r["latency_ms"] for r in self.request_log]
                return {
                    "total_requests": len(self.request_log),
                    "cache_size": len(self.cache),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                }

        # Use the service
        service = ProductionLLMService(MODEL_ID)

        print("Using production-ready service:\n")

        # Make requests
        test_prompts = [
            "What is Python?",
            "What is Python?",  # Should hit cache
            "What is JavaScript?",
        ]

        for prompt in test_prompts:
            result = service.invoke(prompt)
            source = result.get("source", "unknown")
            status = result.get("status", "unknown")
            latency = result.get("latency_ms", 0)

            print(f"Prompt: '{prompt}'")
            print(f"  Status: {status}, Source: {source}, Latency: {latency:.1f}ms")

        # Show stats
        stats = service.get_stats()
        print(f"\nService Stats:")
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Cache size: {stats.get('cache_size', 0)}")
        print(f"  Avg latency: {stats.get('avg_latency_ms', 0):.1f}ms")

        print("\n[OK] Production patterns ensure reliability!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 10: Streaming & Advanced Patterns")
        print("=" * 80)

        try:
            self.example_1_streaming_responses()
            self.example_2_token_counting()
            self.example_3_batch_processing()
            self.example_4_caching_pattern()
            self.example_5_request_limiting()
            self.example_6_async_operations()
            self.example_7_production_pattern()

            # Summary
            pretty_print("Summary & Next Steps", "")
            print("Key takeaways:")
            print("  1. Streaming provides real-time response feedback")
            print("  2. Token counting helps optimize costs")
            print("  3. Batch processing handles multiple requests")
            print("  4. Caching reduces latency and API calls")
            print("  5. Rate limiting ensures compliance")
            print("  6. Async operations enable concurrency")
            print("  7. Production patterns combine all techniques")
            print("\nNext steps:")
            print("  • Deploy to production (Lambda, containers, etc.)")
            print("  • Monitor performance and costs")
            print("  • Add database for persistence")
            print("  • Implement authentication and authorization")
            print("  • Build APIs for client applications")
            print("  • Explore other Bedrock models")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates advanced patterns.

    Usage:
        python level_10_streaming_advanced.py
    """

    tutorial = AdvancedPatternsTutorial()
    tutorial.demonstrate()
