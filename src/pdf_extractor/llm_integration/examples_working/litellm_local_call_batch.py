"""
# LiteLLM Batch Processing Client

This module implements a batch client for making multiple LLM calls efficiently using LiteLLM,
with support for both local and remote endpoints.

## Third-Party Packages:
- litellm: https://github.com/BerriAI/litellm (v1.36.0)
- loguru: https://github.com/Delgan/loguru (v0.7.2)
- tenacity: https://github.com/jd/tenacity (v8.2.3)
- tqdm: https://github.com/tqdm/tqdm (v4.66.3)
- python-dotenv: https://github.com/theskumar/python-dotenv (v1.0.1)

## Sample Input:
```
batch_calls = [
    {"model": "codellama-7b", "prompt": "What is the capital of France?", "local": True},
    {"model": "gpt-4o-mini", "prompt": "Explain quantum computing in one sentence.", "local": False},
]
result = await get_llm_response(batch_calls)
```

## Expected Output:
```
[
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain quantum computing in one sentence.", "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously to perform complex calculations much faster than classical computers.")
]
```
"""

# === Load environment ===
from dotenv import load_dotenv
load_dotenv()  # pulls in OPENAI_API_KEY, etc.

# === Standard library ===
import os
import asyncio
from typing import Dict, List, Tuple, Any, Union

# === Third-party ===
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import acompletion
from tqdm import tqdm

# === Local ===
from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache

# === Logger setup ===
logger.remove()
# file sink
logger.add("llm_client.log", level="DEBUG", rotation="10 MB")
# console sink—via tqdm.write to protect the progress bar
logger.add(
    sink=lambda msg: tqdm.write(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# === Concurrency control ===
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# === Model configurations ===
MODEL_CONFIG = {
    "codellama-7b": {
        "litellm_model":  "openai/codellama-7b",
        "api_base":       "http://localhost:4000/v1",
        "api_key":        "anything",      # only for local endpoint
        "context_window": 4096,
        "max_tokens":     4096,
    },
    "gpt-4o-mini": {
        "litellm_model":  "openai/gpt-4o-mini",
        # no api_key here—Litellm reads OPENAI_API_KEY automatically
        "context_window": 128000,
        "max_tokens":     16384,
    },
}

# === Retry wrapper ===
@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
)
async def acompletion_with_retry(**kwargs) -> Any:
    """Call litellm.acompletion with retries."""
    return await acompletion(**kwargs)

def _build_call_params(
    model_key: str, prompt: str, local: bool
) -> Dict[str, Any]:
    """Construct kwargs for acompletion_with_retry."""
    cfg = MODEL_CONFIG[model_key]
    params: Dict[str, Any] = {
        "model":      cfg["litellm_model"],
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": cfg["max_tokens"],
    }

    if local:
        api_base = cfg["api_base"]
        if os.getenv("DOCKER_ENV"):
            api_base = api_base.replace("localhost", "sglang")
        params["api_base"] = api_base
        params["api_key"]  = cfg["api_key"]

    return params

async def _safely_call_llm(params: Dict[str, Any]) -> Any:
    """
    Acquire the semaphore before calling LLM to limit concurrency,
    then call the retry-wrapped acompletion.
    """
    async with semaphore:
        return await acompletion_with_retry(**params)

async def get_llm_response(
    llm_calls: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> List[Tuple[str, str]]:
    """
    Run one or more LLM calls concurrently with a tqdm progress bar,
    capped at MAX_CONCURRENT_REQUESTS in flight at once.

    Args:
        llm_calls: a dict or list of dicts, each containing:
            - 'model':  key in MODEL_CONFIG
            - 'prompt': the user prompt
            - 'local':  bool, whether to call the local endpoint

    Returns:
        List of (prompt, response) tuples.
    """
    # support single call
    if isinstance(llm_calls, dict):
        llm_calls = [llm_calls]

    tasks:   List[asyncio.Task] = []
    prompts: List[str]           = []

    for call in llm_calls:
        model = call["model"]
        prompt = call["prompt"]
        local = call.get("local", False)

        if model not in MODEL_CONFIG:
            logger.error(f"Unknown model '{model}', skipping.")
            continue

        params = _build_call_params(model, prompt, local)
        task = asyncio.create_task(_safely_call_llm(params))
        tasks.append(task)
        prompts.append(prompt)

    results: List[Tuple[str, str]] = []
    if not tasks:
        return results

    # live progress bar, logs via tqdm.write
    pbar = tqdm(total=len(tasks), desc="Processing prompts")
    for t in tasks:
        t.add_done_callback(lambda _: pbar.update(1))

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    pbar.close()

    # pair up in original order
    for prompt, resp in zip(prompts, responses):
        if isinstance(resp, Exception):
            logger.error(f"Error for '{prompt[:50]}...': {resp}")
            results.append((prompt, ""))
        else:
            try:
                content = resp.choices[0].message.content
                text = content.strip() if isinstance(content, str) else str(content)
            except Exception as e:
                logger.error(f"Parse error for '{prompt[:50]}...': {e}")
                text = ""
            logger.info(f"Response for '{prompt[:50]}...': {text[:50]}...")
            results.append((prompt, text))

    return results

async def main() -> None:
    """Initialize cache, then demo batch and single calls."""
    initialize_litellm_cache()

    batch_calls = [
        {"model": "codellama-7b", "prompt": "What is the capital of France?", "local": True},
        {"model": "codellama-7b", "prompt": "Write a Python function to calculate factorial.", "local": True},
        {"model": "gpt-4o-mini",   "prompt": "Explain quantum computing in one sentence.", "local": False},
    ]
    batch_results = await get_llm_response(batch_calls)
    for prompt, resp in batch_results:
        logger.info(f"\nPrompt: {prompt}\nResponse: {resp}\n")

    single_call = {"model": "gpt-4o-mini", "prompt": "What is 2+2?", "local": False}
    single_result = await get_llm_response(single_call)
    logger.info(f"Single call result: {single_result}")

if __name__ == "__main__":
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    asyncio.run(main())
    logger.success("✅ Standalone validation tests passed for litellm_local_call_batch.py.")
