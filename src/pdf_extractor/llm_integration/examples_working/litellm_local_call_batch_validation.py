"""
# LiteLLM Batch Processing Client with Improved Corpus Validation Logging

This module implements a batch client for making multiple LLM calls efficiently using LiteLLM,
with support for both local and remote endpoints, JSON schema validation, exact answer validation,
fuzzy matching using RapidFuzz, and a gather-timeout so it never hangs indefinitely.

## Third-Party Packages:
- litellm: https://github.com/BerriAI/litellm
- rapidfuzz: https://github.com/maxbachmann/RapidFuzz
- loguru: https://github.com/Delgan/loguru
- tenacity: https://github.com/jd/tenacity
- pydantic: https://docs.pydantic.dev/latest/
- tqdm: https://github.com/tqdm/tqdm

## Sample Input:
Batch of LLM calls with validation criteria:
```python
calls = [
    {
        "model": "codellama-7b",
        "prompt": "What is 2+2?",
        "response_schema": QuestionAnswer,
        "validation": {
            "type": "exact",
            "expected": "4",
            "field": "answer"
        }
    }
]
```

## Expected Output:
List of tuples containing prompts and responses, with validation results logged:
```python
[("What is 2+2?", QuestionAnswer(question="What is 2+2?", answer="4", confidence="high"))]
```
"""

# === Load environment ===
from dotenv import load_dotenv
load_dotenv()  # pulls in OPENAI_API_KEY, etc.

# === Standard library ===
import os
import json
import sys
import asyncio
import logging
import re
import uuid
from typing import Dict, List, Tuple, Any, Union, Optional, Type

# === Third-party ===
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import acompletion
from tqdm import tqdm
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
import litellm  # for suppress_debug_info

# === Suppress LiteLLM internal logs and info dumps ===
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# === Local ===
from pdf_extractor.llm_integration.initialize_litellm_cache import initialize_litellm_cache

# === Configure our logger to drop Provider List lines ===
logger.remove()
logger.add(
    "llm_client.log",
    level="DEBUG",
    rotation="10 MB",
    filter=lambda record: "Provider List:" not in record["message"]
            and "https://docs.litellm.ai/docs/providers" not in record["message"]
)
logger.add(
    sink=lambda msg: tqdm.write(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    filter=lambda record: "Provider List:" not in record["message"]
            and "https://docs.litellm.ai/docs/providers" not in record["message"]
)

# === Schema Definitions ===
class QuestionAnswer(BaseModel):
    question: str = Field(description="The original question asked")
    answer: str = Field(description="The answer to the question - exact only")
    confidence: str = Field(description="Confidence level: high, medium, or low")

class CodeResponse(BaseModel):
    code: str = Field(description="The generated code")
    language: str = Field(description="The programming language")
    explanation: Optional[str] = Field(None, description="How the code works")

# === Concurrency control ===
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# === Global retry settings ===
DEFAULT_MAX_VALIDATION_RETRIES = 3

# === Monkey-patch print to suppress provider lists ===
_original_print = print
def filtered_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "Provider List:" not in msg:
        _original_print(*args, **kwargs)
print = filtered_print

# === Model configurations ===
MODEL_CONFIG = {
    "codellama-7b": {
        "litellm_model":  "openai/codellama-7b",
        "api_base":       "http://localhost:4000/v1",
        "api_key":        "anything",
        "context_window": 4096,
        "max_tokens":     4096,
    },
    "gpt-4o-mini": {
        "litellm_model":  "openai/gpt-4o-mini",
        "context_window": 128000,
        "max_tokens":     16384,
    },
}

# === Retry wrapper ===
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=3))
async def acompletion_with_retry(**kwargs) -> Any:
    return await acompletion(**kwargs)

def _build_call_params(
    model_key: str,
    prompt: str,
    local: bool,
    response_schema: Optional[Type[BaseModel]] = None,
    exact_answer: bool = False
) -> Dict[str, Any]:
    cfg = MODEL_CONFIG[model_key]
    messages = []
    if exact_answer:
        messages.append({
            "role": "system",
            "content": (
                "Provide ONLY the exact answer with no explanation. "
                "E.g., for 'What is 2+2?' respond with '4' only."
            )
        })
    messages.append({"role": "user", "content": prompt})

    params: Dict[str, Any] = {
        "model":      cfg["litellm_model"],
        "messages":   messages,
        "max_tokens": cfg["max_tokens"],
    }
    if response_schema:
        params["response_format"] = response_schema
    if local:
        api_base = cfg["api_base"]
        if os.getenv("DOCKER_ENV"):
            api_base = api_base.replace("localhost", "sglang")
        params["api_base"] = api_base
        params["api_key"]  = cfg["api_key"]
    return params

async def _safely_call_llm(params: Dict[str, Any], task_id: str) -> Any:
    logger.debug(f"[Task {task_id}] → calling LLM @ {params.get('api_base', params['model'])}")
    async with semaphore:
        result = await acompletion_with_retry(**params)
    logger.debug(f"[Task {task_id}] ← LLM returned")
    return result

def highlight_matching_words(text1: str, text2: str) -> Tuple[str, str]:
    """
    Highlight matching words between two texts.
    Returns both texts with HTML-like tags to highlight matching words.
    """
    # Get all words from both texts
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    # Find matching words
    matching_words = set(words1) & set(words2)
    
    # Create highlighted versions
    highlighted1 = text1
    highlighted2 = text2
    
    for word in matching_words:
        # Don't highlight very common words
        if word in {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}:
            continue
            
        # Case insensitive replace with highlighting
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        highlighted1 = pattern.sub(f"[{word}]", highlighted1)
        highlighted2 = pattern.sub(f"[{word}]", highlighted2)
    
    return highlighted1, highlighted2

def validate_corpus_match(response_text: str, corpus: List[str], threshold: int = 75, task_id: str = "") -> Tuple[bool, Dict[str, Any]]:
    """
    Validates if a response text exists with high similarity somewhere in the corpus.
    
    This checks if the content of the response aligns with approved content
    in the corpus - similar to ensuring compliance with technical standards,
    precedent law, or regulatory requirements.
    
    Args:
        response_text: The LLM response to validate
        corpus: List of approved reference paragraphs
        threshold: Minimum similarity percentage required
        task_id: Task identifier for logging
        
    Returns:
        (is_valid, results): Validation result and detailed match information
    """
    response_clean = response_text.strip().lower()
    
    # Log what we're validating
    logger.debug(f"[Task {task_id}] Validating response against corpus of {len(corpus)} paragraphs")
    logger.debug(f"[Task {task_id}] Response text: \"{response_text[:100]}...\"")
    
    # Initialize results
    results = {
        "valid": False,
        "best_score": 0,
        "best_match": "",
        "best_method": "",
        "matching_words": [],
        "missing_words": [],
        "match_details": None
    }
    
    # Check against each paragraph in the corpus
    for i, para in enumerate(corpus):
        para_clean = para.strip().lower()
        
        # Try different fuzzy matching methods
        token_set_score = fuzz.token_set_ratio(response_clean, para_clean)
        token_sort_score = fuzz.token_sort_ratio(response_clean, para_clean)
        partial_score = fuzz.partial_ratio(response_clean, para_clean)
        simple_score = fuzz.ratio(response_clean, para_clean)
        
        # Track all scores
        method_scores = {
            "token_set": token_set_score,
            "token_sort": token_sort_score,
            "partial": partial_score,
            "simple": simple_score
        }
        
        # Find best method for this paragraph
        best_method = max(method_scores.items(), key=lambda x: x[1])
        method_name, score = best_method
        
        # Log paragraph comparison
        logger.debug(f"[Task {task_id}] Corpus paragraph {i+1}: {para[:50]}...")
        logger.debug(f"[Task {task_id}]   Scores: {method_scores}")
        
        # If this is the best match so far
        if score > results["best_score"]:
            results["best_score"] = score
            results["best_match"] = para
            results["best_method"] = method_name
            
            # Find matching and missing keywords
            response_words = set(re.findall(r'\b\w+\b', response_clean))
            para_words = set(re.findall(r'\b\w+\b', para_clean))
            
            # Filter out common words
            common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
            key_response_words = response_words - common_words
            key_para_words = para_words - common_words
            
            results["matching_words"] = sorted(key_response_words & key_para_words)
            results["missing_words"] = sorted(key_para_words - key_response_words)
            
            # Create highlighted versions for visualization
            highlighted_resp, highlighted_para = highlight_matching_words(response_text, para)
            
            # Calculate word overlap for statistics
            results["match_details"] = {
                "highlighted_response": highlighted_resp,
                "highlighted_paragraph": highlighted_para,
                "matching_words_count": len(results["matching_words"]),
                "total_words_paragraph": len(key_para_words),
                "word_overlap_percent": round(len(results["matching_words"]) / len(key_para_words) * 100, 1) 
                                        if key_para_words else 0
            }
    
    # Check if best match exceeds threshold
    results["valid"] = results["best_score"] >= threshold
    
    # Log validation result
    if results["valid"]:
        logger.debug(f"[Task {task_id}] ✅ Corpus validation PASSED: {results['best_score']}% match")
    else:
        logger.debug(f"[Task {task_id}] ❌ Corpus validation FAILED: {results['best_score']}% match (threshold: {threshold}%)")
    
    return results["valid"], results

async def get_llm_response(
    llm_calls: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> List[Tuple[str, Any]]:
    """
    Processes a batch of LLM calls with validation.
    
    For each call, sends the prompt to the specified model, validates the response
    according to the specified validation criteria, and retries if validation fails
    (up to the specified retry limit).
    
    Args:
        llm_calls: Single call or list of calls with model, prompt, and validation info
        
    Returns:
        List of (prompt, response) tuples for each call
    """
    # Ensure llm_calls is a list
    if isinstance(llm_calls, dict):
        llm_calls = [llm_calls]

    # Initialize results list
    results = []
    
    # Generate unique task IDs for this batch
    task_ids = [str(uuid.uuid4())[:8] for _ in range(len(llm_calls))]
    
    # Log batch info
    logger.info(f"Processing batch of {len(llm_calls)} LLM calls with validation")

    # Prepare tasks, prompts, and validation info
    tasks, prompts, schemas, validations, max_retries = [], [], [], [], []
    
    for call_idx, (call, task_id) in enumerate(zip(llm_calls, task_ids)):
        call_id = call_idx + 1
        model = call["model"]
        if model not in MODEL_CONFIG:
            logger.error(f"[Task {task_id}] Unknown model '{model}', skipping call {call_id}.")
            continue
        
        prompt = call["prompt"]
        schema = call.get("response_schema")
        
        # Log what we're about to do
        logger.info(f"[Task {task_id}] Call {call_id}: \"{prompt[:50]}...\" using model {model}")
        
        # Collect validation info
        validation = {}
        if "validation" in call:
            validation = call["validation"]
            logger.info(f"[Task {task_id}] Validation type: {validation.get('type', 'none')}")
            
            # Log validation details
            if validation.get("type") == "exact":
                logger.info(f"[Task {task_id}] Expecting exact answer: \"{validation.get('expected', '')}\"")
            elif validation.get("type") == "list":
                logger.info(f"[Task {task_id}] Expecting answer from list: {validation.get('allowed_values', [])}")
            elif validation.get("type") == "corpus":
                logger.info(f"[Task {task_id}] Corpus validation: {len(validation.get('corpus', []))} paragraphs, " 
                           f"threshold {validation.get('threshold', 75)}%")
        elif "expected_content" in call:
            # Legacy format support
            validation = {
                "type": "exact" if call.get("exact_answer", False) else "fuzzy",
                "expected": call["expected_content"]
            }
            if "allowed_values" in call:
                validation["type"] = "list"
                validation["allowed_values"] = call["allowed_values"]
            elif "corpus" in call:
                validation["type"] = "corpus"
                validation["corpus"] = call["corpus"]
                validation["threshold"] = call.get("threshold", 75)
            
            logger.info(f"[Task {task_id}] Using legacy validation format: {validation['type']}")
        
        # Get per-call max_retries or use default
        retry_limit = call.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)
        
        # Determine if exact answer is needed for system prompt
        exact_answer = validation.get("type") == "exact"
        
        params = _build_call_params(model, prompt, call.get("local", False), schema, exact_answer)
        tasks.append(asyncio.create_task(_safely_call_llm(params, task_id)))
        prompts.append(prompt)
        schemas.append(schema)
        validations.append(validation)
        max_retries.append(retry_limit)

    if not tasks:
        return results

    # Setup progress bar
    pbar = tqdm(total=len(tasks), desc="Processing prompts")
    for t in tasks:
        t.add_done_callback(lambda _: pbar.update(1))

    # Handle timeout (30s)
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30
        )
    except asyncio.TimeoutError:
        logger.error("❌ Timeout waiting for LLM responses (30s)")
        # cancel any still-pending tasks
        responses = []
        for t, task_id in zip(tasks, task_ids):
            if not t.done():
                t.cancel()
                logger.error(f"[Task {task_id}] Cancelled due to timeout")
                responses.append(asyncio.TimeoutError("timed out"))
            else:
                try:
                    responses.append(t.result())
                except Exception as e:
                    logger.error(f"[Task {task_id}] Failed with error: {str(e)}")
                    responses.append(e)
    finally:
        pbar.close()

    validation_results = []
    
    # Process each response
    for idx, (prompt, resp, schema, validation, retry_limit, task_id) in enumerate(
        zip(prompts, responses, schemas, validations, max_retries, task_ids)
    ):
        call_id = idx + 1
        logger.info(f"[Task {task_id}] ===== PROCESSING CALL {call_id} RESPONSE =====")
        
        if isinstance(resp, Exception):
            logger.error(f"[Task {task_id}] Error for call {call_id}: {resp}")
            results.append((prompt, ""))
            validation_results.append(False)
            continue

        # Track retry attempts for this prompt
        retry_count = 0
        validation_passed = False
        final_response = None
        
        # Initialize validation-specific storage
        validation_context = {}

        # Main validation loop - will retry until success or max retries
        while retry_count <= retry_limit:
            # Skip retries if we already passed validation
            if validation_passed:
                break
                
            # If this is a retry, make a new call to the LLM
            if retry_count > 0:
                try:
                    logger.info(f"[Task {task_id}] Retry {retry_count}/{retry_limit} for call {call_id}")
                    
                    # Modify the prompt to improve chances of success
                    retry_prompt = prompt
                    if validation and "type" in validation:
                        val_type = validation["type"]
                        
                        # Add hints based on validation type
                        if val_type == "exact" and "expected" in validation:
                            retry_prompt = f"{prompt}\n\nIMPORTANT: Your answer must be EXACTLY correct. No explanation needed."
                            logger.info(f"[Task {task_id}] Retry prompt modified to enforce exact answer")
                        elif val_type == "list" and "allowed_values" in validation:
                            allowed = validation["allowed_values"]
                            retry_prompt = f"{prompt}\n\nYour answer must be one of these exact values: {', '.join(allowed)}"
                            logger.info(f"[Task {task_id}] Retry prompt modified to include allowed values")
                        elif val_type == "corpus":
                            # Use missing words from the previous validation attempt
                            missing = validation_context.get("missing_words", [])[:5]
                            if missing:
                                retry_prompt = f"{prompt}\n\nMake sure to include these key concepts: {', '.join(missing)}"
                                logger.info(f"[Task {task_id}] Retry prompt modified to include missing concepts: {', '.join(missing)}")
                    
                    logger.info(f"[Task {task_id}] Retry prompt: \"{retry_prompt[:100]}...\"")
                    
                    # Make a new call to the LLM
                    model_key = llm_calls[idx]["model"]
                    call_params = _build_call_params(
                        model_key=model_key,  # Use model_key parameter
                        prompt=retry_prompt,
                        local=llm_calls[idx].get("local", False),
                        response_schema=schema,
                        exact_answer=(validation.get("type") == "exact")
                    )
                    resp = await _safely_call_llm(call_params, task_id)
                except Exception as e:
                    logger.error(f"[Task {task_id}] Error during retry {retry_count} for call {call_id}: {e}")
                    logger.debug(f"[Task {task_id}] Exception details:", exc_info=True)
                    break
            
            # Process structured responses with schemas
            if schema and hasattr(resp.choices[0].message, "parsed"):
                parsed = resp.choices[0].message.parsed
                
                # Default to valid unless validation fails
                validation_passed = True
                
                # Perform validation if configured
                if validation and "type" in validation:
                    val_type = validation["type"]
                    
                    # === Exact validation ===
                    if val_type == "exact" and "expected" in validation:
                        expected = validation["expected"]
                        field = validation.get("field", "answer")
                        
                        # Extract the field to validate
                        if hasattr(parsed, field):
                            actual_value = getattr(parsed, field)
                            
                            if actual_value.strip().lower() == expected.strip().lower():
                                logger.info(f"[Task {task_id}] ✅ Exact validation PASSED for call {call_id}: '{actual_value}'")
                            else:
                                logger.error(f"[Task {task_id}] ❌ Exact validation FAILED for call {call_id}:")
                                logger.error(f"[Task {task_id}]    Field: {field}")
                                logger.error(f"[Task {task_id}]    Expected: '{expected}'")
                                logger.error(f"[Task {task_id}]    Actual: '{actual_value}'")
                                validation_passed = False
                        else:
                            logger.error(f"[Task {task_id}] ❌ Exact validation FAILED for call {call_id}: Field '{field}' not found")
                            validation_passed = False
                    
                    # === List validation ===
                    elif val_type == "list" and "allowed_values" in validation:
                        field = validation.get("field", "answer")
                        allowed = validation["allowed_values"]
                        expected = validation.get("expected")
                        
                        # Extract the field to validate
                        if hasattr(parsed, field):
                            actual_value = getattr(parsed, field)
                            actual_lower = actual_value.strip().lower()
                            allowed_lower = [v.strip().lower() for v in allowed]
                            
                            if actual_lower in allowed_lower:
                                logger.info(f"[Task {task_id}] ✅ List validation PASSED for call {call_id}: '{actual_value}' in allowed values")
                                # If expected value specified, check if it matches
                                if expected and actual_lower != expected.strip().lower():
                                    logger.error(f"[Task {task_id}] ❌ List validation FAILED for call {call_id}:")
                                    logger.error(f"[Task {task_id}]    Found '{actual_value}' in allowed values, but expected '{expected}'")
                                    validation_passed = False
                            else:
                                # Not in list - find closest match
                                best_match = ""
                                best_score = 0
                                for val in allowed:
                                    score = fuzz.ratio(actual_lower, val.lower())
                                    if score > best_score:
                                        best_score = score
                                        best_match = val
                                        
                                logger.error(f"[Task {task_id}] ❌ List validation FAILED for call {call_id}:")
                                logger.error(f"[Task {task_id}]    Field: {field}")
                                logger.error(f"[Task {task_id}]    Value: '{actual_value}'")
                                logger.error(f"[Task {task_id}]    Allowed values: {allowed}")
                                if best_score > 70:
                                    logger.error(f"[Task {task_id}]    Closest match: '{best_match}' ({best_score}% similar)")
                                validation_passed = False
                        else:
                            logger.error(f"[Task {task_id}] ❌ List validation FAILED for call {call_id}: Field '{field}' not found")
                            validation_passed = False
                    
                    # === Corpus validation ===
                    elif val_type == "corpus" and "corpus" in validation:
                        corpus = validation["corpus"]
                        threshold = validation.get("threshold", 75)
                        field = validation.get("field")
                        
                        # Log what we're validating against
                        logger.info(f"[Task {task_id}] Validating against corpus ({len(corpus)} paragraphs)")
                        
                        # Extract text to validate
                        if field and hasattr(parsed, field):
                            text_to_check = getattr(parsed, field)
                        else:
                            # Use all text fields combined
                            text_to_check = " ".join(
                                str(v) for v in parsed.model_dump().values() 
                                if isinstance(v, str)
                            )
                        
                        logger.info(f"[Task {task_id}] Response to validate: \"{text_to_check[:100]}...\"")
                        
                        # Run corpus validation
                        is_valid, result_dict = validate_corpus_match(text_to_check, corpus, threshold, task_id)
                        
                        # Store missing words for potential retry improvements
                        validation_context["missing_words"] = result_dict.get("missing_words", [])
                        
                        if is_valid:
                            logger.info(f"[Task {task_id}] ✅ Corpus validation PASSED for call {call_id}:")
                            logger.info(f"[Task {task_id}]    Score: {result_dict['best_score']:.1f}% match via {result_dict['best_method']}")
                            logger.info(f"[Task {task_id}]    Matching words ({len(result_dict['matching_words'])}): "
                                      f"{', '.join(result_dict['matching_words'][:10])}"
                                      f"{'...' if len(result_dict['matching_words']) > 10 else ''}")
                            
                            if 'match_details' in result_dict:
                                logger.debug(f"[Task {task_id}]    Response: {result_dict['match_details']['highlighted_response']}")
                                logger.debug(f"[Task {task_id}]    Matching paragraph: {result_dict['match_details']['highlighted_paragraph']}")
                        else:
                            # Detailed corpus failure information
                            logger.error(f"[Task {task_id}] ❌ Corpus validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Best score: {result_dict['best_score']:.1f}% (threshold: {threshold}%)")
                            logger.error(f"[Task {task_id}]    Method with highest score: {result_dict['best_method']}")
                            
                            # Show matching and missing words
                            if result_dict["matching_words"]:
                                logger.error(f"[Task {task_id}]    Matching words ({len(result_dict['matching_words'])}): "
                                           f"{', '.join(result_dict['matching_words'][:10])}"
                                           f"{'...' if len(result_dict['matching_words']) > 10 else ''}")
                            if result_dict["missing_words"]:
                                logger.error(f"[Task {task_id}]    Missing key words ({len(result_dict['missing_words'])}): "
                                           f"{', '.join(result_dict['missing_words'][:10])}"
                                           f"{'...' if len(result_dict['missing_words']) > 10 else ''}")
                            
                            # Show highlighted text for comparison
                            if 'match_details' in result_dict:
                                stats = result_dict['match_details']
                                logger.error(f"[Task {task_id}]    Word overlap: {stats['matching_words_count']}/{stats['total_words_paragraph']} "
                                           f"words ({stats['word_overlap_percent']}%)")
                                logger.error(f"[Task {task_id}]    Response: {stats['highlighted_response']}")
                                logger.error(f"[Task {task_id}]    Best paragraph: {stats['highlighted_paragraph']}")
                                
                                # Suggestion for improving the match
                                if result_dict['best_score'] > threshold * 0.7:  # If we're close
                                    words_to_add = result_dict["missing_words"][:5]
                                    if words_to_add:
                                        logger.warning(f"[Task {task_id}]    ⚠️ To improve match: Add these key words: {', '.join(words_to_add)}")
                                
                            validation_passed = False
                    
                    # === Fuzzy validation ===
                    elif val_type == "fuzzy" and "expected" in validation:
                        expected = validation["expected"]
                        field = validation.get("field")
                        threshold = validation.get("threshold", 97)
                        
                        # Extract text to validate
                        if field and hasattr(parsed, field):
                            actual_value = getattr(parsed, field)
                        else:
                            # Use all text fields combined
                            actual_value = " ".join(
                                str(v) for v in parsed.model_dump().values() 
                                if isinstance(v, str)
                            )
                        
                        # Try different fuzzy matching approaches
                        simple_score = fuzz.ratio(expected.lower(), actual_value.lower())
                        partial_score = fuzz.partial_ratio(expected.lower(), actual_value.lower())
                        token_sort_score = fuzz.token_sort_ratio(expected.lower(), actual_value.lower())
                        
                        # Use the best score from any method
                        score = max(simple_score, partial_score, token_sort_score)
                        best_method = ["simple", "partial", "token_sort"][
                            [simple_score, partial_score, token_sort_score].index(score)
                        ]
                        
                        if score >= threshold:
                            logger.info(f"[Task {task_id}] ✅ Fuzzy validation PASSED for call {call_id}: {score:.1f}% match via {best_method}")
                        else:
                            logger.error(f"[Task {task_id}] ❌ Fuzzy validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Best match score: {score:.1f}% (threshold: {threshold}%)")
                            logger.error(f"[Task {task_id}]    Method scores: simple: {simple_score:.1f}%, partial: {partial_score:.1f}%, "
                                        f"token_sort: {token_sort_score:.1f}%")
                            logger.error(f"[Task {task_id}]    Expected: '{expected}'")
                            logger.error(f"[Task {task_id}]    Actual: '{actual_value[:100]}...'")
                            
                            # Highlight matching words
                            highlighted_expected, highlighted_actual = highlight_matching_words(expected, actual_value[:100])
                            logger.error(f"[Task {task_id}]    Expected with highlights: {highlighted_expected}")
                            logger.error(f"[Task {task_id}]    Actual with highlights: {highlighted_actual}")
                            validation_passed = False
                
                # If validation passed or we've reached max retries, store the final response
                if validation_passed or retry_count >= retry_limit:
                    final_response = parsed
                    
                    # If we've reached max retries and still failed, log a helpful message
                    if not validation_passed and retry_count >= retry_limit:
                        logger.error(f"[Task {task_id}] ⛔ Max retries ({retry_limit}) reached for call {call_id}. Validation still failing.")
                        logger.error(f"[Task {task_id}]    Consider adjusting your prompt or validation criteria.")
                
                # Log the parsed response
                if hasattr(parsed, "model_dump"):
                    logger.info(f"[Task {task_id}] Response for call {call_id}: {parsed.model_dump()}")
                else:
                    logger.info(f"[Task {task_id}] Response for call {call_id}: {parsed}")
            
            # Process text responses (no schemas or parsing failed)
            else:
                content = resp.choices[0].message.content
                text = content.strip() if isinstance(content, str) else str(content)
                
                logger.info(f"[Task {task_id}] Text response for call {call_id}: \"{text[:100]}...\"")
                
                # Default to valid unless validation fails
                validation_passed = True
                
                # Perform validation if configured
                if validation and "type" in validation:
                    val_type = validation["type"]
                    
                    # === Exact validation ===
                    if val_type == "exact" and "expected" in validation:
                        expected = validation["expected"]
                        
                        # For text responses, try to extract JSON if schema expected
                        if schema:
                            try:
                                data = json.loads(text)
                                field = validation.get("field", "answer")
                                if field in data:
                                    actual_value = data[field]
                                else:
                                    actual_value = text
                            except json.JSONDecodeError:
                                actual_value = text
                        else:
                            actual_value = text
                        
                        if actual_value.strip().lower() == expected.strip().lower():
                            logger.info(f"[Task {task_id}] ✅ Exact validation PASSED for call {call_id}: '{actual_value}'")
                        else:
                            logger.error(f"[Task {task_id}] ❌ Exact validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Expected: '{expected}'")
                            logger.error(f"[Task {task_id}]    Actual: '{actual_value}'")
                            validation_passed = False
                    
                    # === List validation ===
                    elif val_type == "list" and "allowed_values" in validation:
                        allowed = validation["allowed_values"]
                        expected = validation.get("expected")
                        
                        # For text responses, try to extract JSON if schema expected
                        if schema:
                            try:
                                data = json.loads(text)
                                field = validation.get("field", "answer")
                                if field in data:
                                    actual_value = data[field]
                                else:
                                    actual_value = text
                            except json.JSONDecodeError:
                                actual_value = text
                        else:
                            actual_value = text
                        
                        actual_lower = actual_value.strip().lower()
                        allowed_lower = [v.strip().lower() for v in allowed]
                        
                        if actual_lower in allowed_lower:
                            logger.info(f"[Task {task_id}] ✅ List validation PASSED for call {call_id}: '{actual_value}' in allowed values")
                            # If expected value specified, check if it matches
                            if expected and actual_lower != expected.strip().lower():
                                logger.error(f"[Task {task_id}] ❌ List validation FAILED for call {call_id}:")
                                logger.error(f"[Task {task_id}]    Found '{actual_value}' in allowed values, but expected '{expected}'")
                                validation_passed = False
                        else:
                            # Not in list - find closest match
                            best_match = ""
                            best_score = 0
                            for val in allowed:
                                score = fuzz.ratio(actual_lower, val.lower())
                                if score > best_score:
                                    best_score = score
                                    best_match = val
                                    
                            logger.error(f"[Task {task_id}] ❌ List validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Value: '{actual_value}'")
                            logger.error(f"[Task {task_id}]    Allowed values: {allowed}")
                            if best_score > 70:
                                logger.error(f"[Task {task_id}]    Closest match: '{best_match}' ({best_score}% similar)")
                            validation_passed = False
                    
                    # === Corpus validation ===
                    elif val_type == "corpus" and "corpus" in validation:
                        corpus = validation["corpus"]
                        threshold = validation.get("threshold", 75)
                        
                        # Log what we're validating against
                        logger.info(f"[Task {task_id}] Validating text response against corpus ({len(corpus)} paragraphs)")
                        
                        # Run corpus validation
                        is_valid, result_dict = validate_corpus_match(text, corpus, threshold, task_id)
                        
                        # Store missing words for potential retry improvements
                        validation_context["missing_words"] = result_dict.get("missing_words", [])
                        
                        if is_valid:
                            logger.info(f"[Task {task_id}] ✅ Corpus validation PASSED for call {call_id}:")
                            logger.info(f"[Task {task_id}]    Score: {result_dict['best_score']:.1f}% match via {result_dict['best_method']}")
                            logger.info(f"[Task {task_id}]    Matching words ({len(result_dict['matching_words'])}): "
                                      f"{', '.join(result_dict['matching_words'][:10])}"
                                      f"{'...' if len(result_dict['matching_words']) > 10 else ''}")
                        else:
                            # Detailed corpus failure information
                            logger.error(f"[Task {task_id}] ❌ Corpus validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Best score: {result_dict['best_score']:.1f}% (threshold: {threshold}%)")
                            logger.error(f"[Task {task_id}]    Method with highest score: {result_dict['best_method']}")
                            
                            # Show matching and missing words
                            if result_dict["matching_words"]:
                                logger.error(f"[Task {task_id}]    Matching words ({len(result_dict['matching_words'])}): "
                                           f"{', '.join(result_dict['matching_words'][:10])}"
                                           f"{'...' if len(result_dict['matching_words']) > 10 else ''}")
                            if result_dict["missing_words"]:
                                logger.error(f"[Task {task_id}]    Missing key words ({len(result_dict['missing_words'])}): "
                                           f"{', '.join(result_dict['missing_words'][:10])}"
                                           f"{'...' if len(result_dict['missing_words']) > 10 else ''}")
                            
                            # Show highlighted text for comparison
                            if 'match_details' in result_dict:
                                stats = result_dict['match_details']
                                logger.error(f"[Task {task_id}]    Word overlap: {stats['matching_words_count']}/{stats['total_words_paragraph']} "
                                           f"words ({stats['word_overlap_percent']}%)")
                                logger.error(f"[Task {task_id}]    Response: {stats['highlighted_response']}")
                                logger.error(f"[Task {task_id}]    Best paragraph: {stats['highlighted_paragraph']}")
                                
                                # Suggestion for improving the match
                                if result_dict['best_score'] > threshold * 0.7:  # If we're close
                                    words_to_add = result_dict["missing_words"][:5]
                                    if words_to_add:
                                        logger.warning(f"[Task {task_id}]    ⚠️ To improve match: Add these key words: {', '.join(words_to_add)}")
                            
                            validation_passed = False
                    
                    # === Fuzzy validation ===
                    elif val_type == "fuzzy" and "expected" in validation:
                        expected = validation["expected"]
                        threshold = validation.get("threshold", 97)
                        
                        # Try different fuzzy matching approaches
                        simple_score = fuzz.ratio(expected.lower(), text.lower())
                        partial_score = fuzz.partial_ratio(expected.lower(), text.lower())
                        token_sort_score = fuzz.token_sort_ratio(expected.lower(), text.lower())
                        
                        # Use the best score from any method
                        score = max(simple_score, partial_score, token_sort_score)
                        best_method = ["simple", "partial", "token_sort"][
                            [simple_score, partial_score, token_sort_score].index(score)
                        ]
                        
                        if score >= threshold:
                            logger.info(f"[Task {task_id}] ✅ Fuzzy validation PASSED for call {call_id}: {score:.1f}% match via {best_method}")
                        else:
                            logger.error(f"[Task {task_id}] ❌ Fuzzy validation FAILED for call {call_id}:")
                            logger.error(f"[Task {task_id}]    Best match score: {score:.1f}% (threshold: {threshold}%)")
                            logger.error(f"[Task {task_id}]    Method scores: simple: {simple_score:.1f}%, partial: {partial_score:.1f}%, "
                                        f"token_sort: {token_sort_score:.1f}%")
                            logger.error(f"[Task {task_id}]    Expected: '{expected}'")
                            logger.error(f"[Task {task_id}]    Actual: '{text[:100]}...'")
                            
                            # Highlight matching words
                            highlighted_expected, highlighted_actual = highlight_matching_words(expected, text[:100])
                            logger.error(f"[Task {task_id}]    Expected with highlights: {highlighted_expected}")
                            logger.error(f"[Task {task_id}]    Actual with highlights: {highlighted_actual}")
                            validation_passed = False
                
                # If validation passed or we've reached max retries, store the final response
                if validation_passed or retry_count >= retry_limit:
                    final_response = text
                    
                    # If we've reached max retries and still failed, log a helpful message
                    if not validation_passed and retry_count >= retry_limit:
                        logger.error(f"[Task {task_id}] ⛔ Max retries ({retry_limit}) reached for call {call_id}. Validation still failing.")
                        logger.error(f"[Task {task_id}]    Consider adjusting your prompt or validation criteria.")
            
            # If validation passed, break out of retry loop
            if validation_passed:
                logger.info(f"[Task {task_id}] ✅ Validation PASSED for call {call_id}")
                break
            
            # Increment retry count and continue if not exceeded max
            retry_count += 1
        
        # Record validation result and add response to results list
        validation_results.append(validation_passed)
        results.append((prompt, final_response))
        
        # Log completion of this call
        logger.info(f"[Task {task_id}] ===== COMPLETED CALL {call_id} =====")

    # Validation summary
    logger.info("===== VALIDATION SUMMARY =====")
    
    if validation_results:
        passed = sum(1 for r in validation_results if r)
        total = len(validation_results)
        if all(validation_results):
            logger.success(f"✅ All {total} validation checks passed")
        else:
            logger.warning(f"⚠️ Validation: {passed}/{total} checks passed")
            
            # List failed task IDs
            failed_tasks = [task_id for i, (task_id, passed) in enumerate(zip(task_ids, validation_results)) if not passed]
            if failed_tasks:
                logger.warning(f"Failed task IDs: {', '.join(failed_tasks)}")
    
    return results

async def main() -> None:
    """
    Run validation tests for different validation types.
    """
    logger.info("\n========= LITELLM BATCH VALIDATION DEMO =========\n")
    
    # silence litellm logs during cache init
    for name in logging.root.manager.loggerDict:
        if name.startswith("litellm"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    # suppress stdout from cache init
    orig, sys.stdout = sys.stdout, type("D", (), {"write":lambda *_:None,"flush":lambda *_:None})()
    initialize_litellm_cache()
    sys.stdout = orig

    litellm.enable_json_schema_validation = True

    # Run exact validation test
    logger.info("\n========== EXACT VALIDATION TEST ==========")
    logger.info("Testing if LLM correctly answers '4' to '2+2'")
    logger.info("===========================================\n")
    
    test_call = {
        "model": "codellama-7b",
        "prompt": "What is 2+2? Provide only the number.",
        "local": True,
        "response_schema": QuestionAnswer,
        "validation": {
            "type": "exact",
            "expected": "4",
            "field": "answer"
        },
        "max_validation_retries": 2  # Override default retry limit
    }
    
    results = await get_llm_response(test_call)
    if not results:
        logger.error("❌ Exact validation test failed")
        sys.exit(1)
    
    # Run list validation test
    logger.info("\n========== LIST VALIDATION TEST ==========")
    logger.info("Testing if LLM correctly identifies 'Paris' as the capital of France")
    logger.info("===========================================\n")
    
    list_test = {
        "model": "codellama-7b",
        "prompt": "What is the capital of France? Provide only the name.",
        "local": True,
        "response_schema": QuestionAnswer,
        "validation": {
            "type": "list",
            "allowed_values": ["Madrid", "Berlin", "Paris", "Rome", "London"],
            "expected": "Paris",
            "field": "answer"
        }
    }
    
    results = await get_llm_response(list_test)
    if not results:
        logger.error("❌ List validation test failed")
        sys.exit(1)
    
    # Run corpus validation test
    logger.info("\n========== CORPUS VALIDATION TEST ==========")
    logger.info("Testing if LLM explanation of a qubit matches the quantum computing corpus")
    logger.info("=============================================\n")
    
    # Example corpus with quantum computing content
    corpus = [
        "Quantum computing uses the principles of quantum mechanics to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.",
        "The power of quantum computing comes from the ability to use quantum phenomena such as superposition and entanglement.",
        "A qubit can exist in multiple states simultaneously due to superposition, unlike a classical bit which can only be 0 or 1.",
        "Quantum entanglement allows qubits to be correlated with each other, no matter how far apart they are physically."
    ]
    
    corpus_test = {
        "model": "codellama-7b",
        "prompt": "Explain what a qubit is in one sentence.",
        "local": True,
        "validation": {
            "type": "corpus",
            "corpus": corpus,
            "threshold": 75
        }
    }
    
    await get_llm_response(corpus_test)
    
    # Retry with improved prompt for corpus validation
    logger.info("\n========== IMPROVED CORPUS VALIDATION TEST ==========")
    logger.info("Testing with improved prompt that includes key concepts")
    logger.info("======================================================\n")
    
    improved_corpus_test = {
        "model": "codellama-7b",
        "prompt": "Explain what a qubit is in one sentence. Be sure to mention superposition and the difference from classical bits.",
        "local": True,
        "validation": {
            "type": "corpus",
            "corpus": corpus,
            "threshold": 75
        }
    }
    
    await get_llm_response(improved_corpus_test)
    
    # Run fuzzy validation test
    logger.info("\n========== FUZZY VALIDATION TEST ==========")
    logger.info("Testing if LLM's factorial function contains expected code pattern")
    logger.info("============================================\n")
    
    fuzzy_test = {
        "model": "codellama-7b",
        "prompt": "Write a Python function to calculate factorial.",
        "local": True,
        "response_schema": CodeResponse,
        "validation": {
            "type": "fuzzy",
            "expected": "def factorial",
            "field": "code",
            "threshold": 80
        }
    }
    
    await get_llm_response(fuzzy_test)

    logger.success("\n✅ All validation tests and demo runs completed successfully")
    sys.exit(0)

# Create a simple log processor that can be used to group and analyze logs by task ID
def process_logs(log_file_path):
    """
    Process log file to group entries by task ID.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Dictionary mapping task IDs to lists of log entries
    """
    task_logs = {}
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Extract timestamp and task ID
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+) \| \[Task (\w+)\]', line)
                if match:
                    timestamp, level, task_id = match.groups()
                    if task_id not in task_logs:
                        task_logs[task_id] = []
                    
                    task_logs[task_id].append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': line.strip()
                    })
    except Exception as e:
        print(f"Error processing log file: {e}")
    
    return task_logs

def generate_task_report(task_logs, output_dir='task_reports'):
    """
    Generate a report file for each task.
    
    Args:
        task_logs: Dictionary mapping task IDs to lists of log entries
        output_dir: Directory to write report files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for task_id, logs in task_logs.items():
        # Sort logs by timestamp
        logs.sort(key=lambda x: x['timestamp'])
        
        # Write report file
        with open(f"{output_dir}/task_{task_id}_report.txt", 'w') as f:
            f.write(f"==== TASK {task_id} LOG REPORT ====\n\n")
            
            # Count log entries by level
            level_counts = {}
            for log in logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Write summary
            f.write("Summary:\n")
            f.write(f"  Total log entries: {len(logs)}\n")
            for level, count in level_counts.items():
                f.write(f"  {level}: {count}\n")
            f.write("\n")
            
            # Write chronological log entries
            f.write("Log Entries (Chronological):\n")
            for log in logs:
                f.write(f"{log['timestamp']} | {log['level']} | {log['message']}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    for name in ["litellm","httpx","httpcore"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate=False

    asyncio.run(main())
    
    # Uncomment to process logs after running the main function
    # task_logs = process_logs("llm_client.log")
    # generate_task_report(task_logs)