"""
Initializes LiteLLM Caching Configuration.

This module sets up LiteLLM's caching mechanism. It attempts to configure
Redis as the primary cache backend. If Redis is unavailable or fails connection
tests, it falls back to using LiteLLM's built-in in-memory cache. Includes
a test function to verify cache functionality.

Relevant Documentation:
- LiteLLM Caching: https://docs.litellm.ai/docs/proxy/caching
- Redis Python Client: https://redis.io/docs/clients/python/
- Project Caching Notes: ../../repo_docs/caching_strategy.md (Placeholder)

Input/Output:
- Input: Environment variables for Redis connection (optional).
- Output: Configures LiteLLM global cache settings. Logs status messages.
- The `test_litellm_cache` function demonstrates usage by making cached calls.
"""

# ==============================================================================
# !!! WARNING - DO NOT MODIFY THIS FILE !!!
# ==============================================================================
# This is a core functionality file that other parts of the system depend on.
# Changing this file (especially its sync/async nature) will break multiple dependent systems.
# Any changes here will cascade into test failures across the codebase.
#
# If you think you need to modify this file:
# 1. DON'T. The synchronous implementation is intentional
# 2. The caching system is working as designed
# 3. Test files should adapt to this implementation, not vice versa
# 4. Consult LESSONS_LEARNED.md about not breaking working code
# ==============================================================================


import litellm
import os
import redis
import sys # Import sys for exit codes
from loguru import logger
from litellm.caching.caching import Cache as LiteLLMCache, LiteLLMCacheType # Import Cache and Type
from typing import Any # Import Any for type hint
# Import the utility function using absolute path
try:
    from pdf_extractor.llm_integration.utils.log_utils import truncate_large_value
except ImportError:
    logger.warning("Could not import truncate_large_value. Logs might be verbose.")
    # Define a dummy function with compatible signature if import fails
    def truncate_large_value(value: Any, *args: Any, **kwargs: Any) -> Any: return value

# load_env_file() # Removed - Docker Compose handles .env loading via env_file


def initialize_litellm_cache() -> None:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_password = os.getenv("REDIS_PASSWORD", None) # Assuming password might be needed

    try:
        logger.debug(f"Starting LiteLLM cache initialization (Redis target: {redis_host}:{redis_port})...")

        # Test Redis connection before enabling caching
        logger.debug("Testing Redis connection...")
        test_redis = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, socket_timeout=2, decode_responses=True # Added decode_responses for easier debugging if needed
        )
        if not test_redis.ping():
            raise ConnectionError(f"Redis is not responding at {redis_host}:{redis_port}.")

        # Verify Redis is empty or log existing keys
        keys = test_redis.keys("*")
        if keys:
            # Use the truncate utility for logging keys
            logger.debug(f"Existing Redis keys: {truncate_large_value(keys)}")
        else:
            logger.debug("Redis cache is empty")

        # Set up LiteLLM cache with debug logging
        logger.debug("Configuring LiteLLM Redis cache...")
        litellm.cache = LiteLLMCache( # Use imported Cache
            type=LiteLLMCacheType.REDIS, # Use Enum/Type
            host=redis_host,
            port=str(redis_port), # Ensure port is a string
            password=redis_password,
            supported_call_types=["acompletion", "completion"],
            ttl=60 * 60 * 24 * 2,  # 2 days
        )

        # Enable caching and verify
        logger.debug("Enabling LiteLLM cache...")
        litellm.enable_cache()

        # Set debug logging for LiteLLM
        os.environ["LITELLM_LOG"] = "DEBUG"

        # Verify cache configuration
        logger.debug(
            f"LiteLLM cache config: {litellm.cache.__dict__ if hasattr(litellm.cache, '__dict__') else 'No cache config available'}"
        )
        logger.info("✅ Redis caching enabled on localhost:6379")

        # Try a test set/get to verify Redis is working
        try:
            test_key = "litellm_cache_test"
            test_redis.set(test_key, "test_value", ex=60)
            result = test_redis.get(test_key)
            logger.debug(f"Redis test write/read successful: {result == 'test_value'}")
            test_redis.delete(test_key)
        except Exception as e:
            logger.warning(f"Redis test write/read failed: {e}")

    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(
            f"⚠️ Redis connection failed: {e}. Falling back to in-memory caching."
        )
        # Fall back to in-memory caching if Redis is unavailable
        logger.debug("Configuring in-memory cache fallback...")
        litellm.cache = LiteLLMCache(type=LiteLLMCacheType.LOCAL) # Use Enum/Type
        litellm.enable_cache()
        logger.debug("In-memory cache enabled")


def test_litellm_cache():
    """Test the LiteLLM cache functionality with a sample completion call"""
    initialize_litellm_cache()

    try:
        # Test the cache with a simple completion call
        test_messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]  # Make sure it's >1024 tokens
        logger.info("Testing cache with completion call...")

        # First call should miss cache
        response1 = litellm.completion(
            model="gpt-4o-mini",
            messages=test_messages,
            cache={"no-cache": False},
        )
        # Access usage and cache_hit using getattr for safety
        usage1 = getattr(response1, 'usage', 'N/A') # Default to 'N/A' if not found
        # Access _hidden_params safely, then get cache_hit
        hidden_params1 = getattr(response1, '_hidden_params', {})
        cache_hit1 = hidden_params1.get("cache_hit")
        logger.info(f"First call usage: {usage1}")
        if cache_hit1 is not None:
             logger.info(f"Response 1: Cache hit: {cache_hit1}")

        # Second call should hit cache
        response2 = litellm.completion(
            model="gpt-4o-mini",
            messages=test_messages,
            cache={"no-cache": False},
        )
        usage2 = getattr(response2, 'usage', 'N/A') # Default to 'N/A' if not found
        # Access _hidden_params safely, then get cache_hit
        hidden_params2 = getattr(response2, '_hidden_params', {})
        cache_hit2 = hidden_params2.get("cache_hit")
        logger.info(f"Second call usage: {usage2}")
        if cache_hit2 is not None:
             logger.info(f"Response 2: Cache hit: {cache_hit2}")

    except Exception as e:
        logger.error(f"Cache test failed with error: {e}")
        raise


if __name__ == "__main__":
    logger.info("--- Running Standalone Validation for initialize_litellm_cache.py ---")
    try:
        test_litellm_cache()
        logger.success("✅ Standalone validation tests passed for initialize_litellm_cache.py.")
        print("\n✅ VALIDATION COMPLETE - LiteLLM cache initialization and testing verified.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Standalone validation failed for initialize_litellm_cache.py: {e}", exc_info=True)
        print("\n❌ VALIDATION FAILED - LiteLLM cache initialization or testing failed.")
        sys.exit(1)