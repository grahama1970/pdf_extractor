# LiteLLM Batch Processing Client with Improved Corpus Validation Logging and Code Execution Validation
#
# This module implements a batch client for making multiple LLM calls efficiently using LiteLLM,
# with support for both local and remote endpoints, JSON schema validation, exact answer validation,
# fuzzy matching using RapidFuzz, code execution validation, and a gather-timeout so it never hangs indefinitely.

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

# === Local ===
from pdf_extractor.llm_client.initialize_litellm_cache import initialize_litellm_cache
from pdf_extractor.llm_client.text_utils import highlight_matching_words
from pdf_extractor.llm_client.validators.corpus_validator import validate_corpus_match
from pdf_extractor.llm_client.validators.code_validator import validate_code_execution, extract_code_from_text
from pdf_extractor.llm_client.schema_models import QuestionAnswer, CodeResponse

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

# === Concurrency control ===
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# === Global retry settings ===
DEFAULT_MAX_VALIDATION_RETRIES = 3

# === Monkey-patch print to suppress provider lists ===
_original_print = print
from typing import Dict, List, Tuple, Any, Union, Optional, Type, Sequence # Add Sequence

def filtered_print(*args: Sequence[Any], **kwargs: Any): # Add : Any hint for kwargs
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
async def acompletion_with_retry(**kwargs: Any) -> Any: # Add : Any hint for kwargs
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
            elif validation.get("type") == "code_execution":
                logger.info(f"[Task {task_id}] Code execution validation: lang={validation.get('language', 'python')}, " 
                           f"timeout={validation.get('timeout', 60)}s")
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
        # Timeout set to 30 seconds
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30
        )
    except asyncio.TimeoutError:
        logger.error("❌ Timeout waiting for LLM responses (30s)")
        # Build responses list, handling cancellations gracefully
        responses = []
        for t, task_id in zip(tasks, task_ids):
            if not t.done():
                t.cancel()
                logger.error(f"[Task {task_id}] Cancelled due to timeout")
                # Append the TimeoutError directly for timed-out tasks
                responses.append(asyncio.TimeoutError(f"Task {task_id} timed out"))
            else:
                try:
                    # For completed tasks, get the result or exception
                    result_or_exc = t.result()
                    responses.append(result_or_exc)
                except asyncio.CancelledError:
                    # If task was cancelled *during* result retrieval (less likely but possible)
                    logger.error(f"[Task {task_id}] Was cancelled during result retrieval after timeout handling.")
                    responses.append(asyncio.CancelledError(f"Task {task_id} cancelled"))
                except Exception as e:
                    # Capture other exceptions from completed tasks
                    logger.error(f"[Task {task_id}] Completed with error: {str(e)}")
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
                    
                    # Add code execution error feedback if applicable
                    if "execution_error" in validation_context and validation_context["execution_error"]:
                        error_msg = validation_context["execution_error"]
                        retry_prompt = f"{prompt}\n\nThe previous code had an error: {error_msg}\nPlease fix the code to make it work properly."
                        logger.info(f"[Task {task_id}] Retry prompt includes code execution error feedback")
                    
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
                    
                    # === Code Execution validation ===
                    elif val_type == "code_execution":
                        language = validation.get("language", "python")
                        timeout = validation.get("timeout", 60)
                        max_memory_mb = validation.get("max_memory_mb", 2048)
                        expected_output = validation.get("expected_output")
                        expected_exit_code = validation.get("expected_exit_code", 0)
                        
                        # Extract code to execute
                        field = validation.get("field", "code")
                        if field and hasattr(parsed, field):
                            code_to_execute = getattr(parsed, field)
                        else:
                            # If no specific field or field not found, assume the entire response is code
                            code_to_execute = extract_code_from_text(" ".join(
                                str(v) for v in parsed.model_dump().values() 
                                if isinstance(v, str)
                            ))
                        
                        if not code_to_execute:
                            logger.error(f"[Task {task_id}] ❌ Code execution validation FAILED for call {call_id}: No code found")
                            validation_passed = False
                            continue
                            
                        # Run code execution validation
                        is_valid, execution_result = validate_code_execution(
                            code=code_to_execute,
                            task_id=task_id,
                            call_id=call_id,
                            language=language,
                            timeout=timeout,
                            max_memory_mb=max_memory_mb
                        )
                        
                        # Store execution error for potential retry improvements
                        if not is_valid:
                            validation_context["execution_error"] = execution_result["stderr"]
                        
                        # If expected output is specified, check if it's in the actual output
                        output_valid = True
                        if expected_output and is_valid:
                            output_valid = expected_output in execution_result["stdout"]
                            if not output_valid:
                                logger.error(f"[Task {task_id}] ❌ Expected output not found in execution result:")
                                logger.error(f"[Task {task_id}]    Expected: \"{expected_output}\"")
                                logger.error(f"[Task {task_id}]    Actual: \"{execution_result['stdout'][:100]}\"")
                        
                        # Final validation result combines execution success and output check
                        validation_passed = is_valid and output_valid
                    
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
                            
                # === Automatic code execution validation for 'code' fields ===
                if validation_passed and "field" in validation and validation["field"] == "code" and validation["type"] != "code_execution":
                    field = validation["field"]
                    if hasattr(parsed, field):
                        code_to_execute = getattr(parsed, field)
                        
                        if code_to_execute:
                            logger.info(f"[Task {task_id}] Field 'code' detected, running automatic code execution validation")
                            execution_valid, execution_result = validate_code_execution(
                                code=code_to_execute,
                                task_id=task_id,
                                call_id=call_id
                            )
                            
                            # If code execution fails, mark validation as failed
                            if not execution_valid:
                                validation_passed = False
                                logger.error(f"[Task {task_id}] ❌ Automatic code execution validation FAILED")
                                
                                # Store execution error for retry improvement
                                validation_context["execution_error"] = execution_result["stderr"]
                
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
                    
                    # === Code Execution validation ===
                    elif val_type == "code_execution":
                        language = validation.get("language", "python")
                        timeout = validation.get("timeout", 60)
                        max_memory_mb = validation.get("max_memory_mb", 2048)
                        expected_output = validation.get("expected_output")
                        
                        # Extract code from text response
                        code_to_execute = extract_code_from_text(text)
                        
                        if not code_to_execute:
                            logger.error(f"[Task {task_id}] ❌ Code execution validation FAILED for call {call_id}: No code found")
                            validation_passed = False
                            continue
                            
                        # Run code execution validation
                        is_valid, execution_result = validate_code_execution(
                            code=code_to_execute,
                            task_id=task_id,
                            call_id=call_id,
                            language=language,
                            timeout=timeout,
                            max_memory_mb=max_memory_mb
                        )
                        
                        # Store execution error for potential retry improvements
                        if not is_valid:
                            validation_context["execution_error"] = execution_result["stderr"]
                        
                        # If expected output is specified, check if it's in the actual output
                        output_valid = True
                        if expected_output and is_valid:
                            output_valid = expected_output in execution_result["stdout"]
                            if not output_valid:
                                logger.error(f"[Task {task_id}] ❌ Expected output not found in execution result:")
                                logger.error(f"[Task {task_id}]    Expected: \"{expected_output}\"")
                                logger.error(f"[Task {task_id}]    Actual: \"{execution_result['stdout'][:100]}\"")
                        
                        # Final validation result combines execution success and output check
                        validation_passed = is_valid and output_valid
                    
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
                
                # === Automatic code execution validation for 'code' fields ===
                if validation_passed and "field" in validation and validation["field"] == "code" and validation["type"] != "code_execution":
                    # Try to extract code from the text response
                    code_to_execute = extract_code_from_text(text)
                    
                    if code_to_execute:
                        logger.info(f"[Task {task_id}] Field 'code' detected, running automatic code execution validation")
                        execution_valid, execution_result = validate_code_execution(
                            code=code_to_execute,
                            task_id=task_id,
                            call_id=call_id
                        )
                        
                        # If code execution fails, mark validation as failed
                        if not execution_valid:
                            validation_passed = False
                            logger.error(f"[Task {task_id}] ❌ Automatic code execution validation FAILED")
                            
                            # Store execution error for retry improvement
                            validation_context["execution_error"] = execution_result["stderr"]
                
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

if __name__ == "__main__":
    import asyncio

    async def run_test():
        tests_passed_count = 0
        tests_failed_count = 0
        total_tests = 1
        test_passed = False
        
        print("--- Running Validation Client Standalone Test ---")
        logger.info("Attempting a simple real LLM call with exact validation...")
        
        # Simple test case using a real model (ensure API key is available in env)
        # Using gpt-4o-mini as it's generally available and fast
        test_call = {
            "model": "gpt-4o-mini",
            "prompt": "Respond exactly with the word 'pong' and nothing else.",
            "validation": {
                "type": "exact",
                "expected": "pong"
            }
        }
        
        expected_response = "pong"
        actual_response_content = None
        
        try:
            # Initialize cache (important for the client)
            initialize_litellm_cache()
            
            # Make the actual call using the client function
            results = await get_llm_response(test_call)
            
            # --- Validation Logic ---
            if results and len(results) == 1:
                prompt, response_obj = results[0]
                # The response object might be a string or a Pydantic model depending on validation
                # For exact match on text, it should return the string
                if isinstance(response_obj, str):
                     actual_response_content = response_obj.strip()
                     if actual_response_content.lower() == expected_response.lower():
                         tests_passed_count += 1
                         test_passed = True
                         print(f"✅ Test 'simple_exact_match': PASSED (Got '{actual_response_content}')")
                     else:
                         tests_failed_count += 1
                         print(f"❌ Test 'simple_exact_match': FAILED - Response mismatch")
                         print(f"   Expected: '{expected_response}'")
                         print(f"   Got: '{actual_response_content}'")
                else:
                    tests_failed_count += 1
                    print(f"❌ Test 'simple_exact_match': FAILED - Unexpected response type")
                    print(f"   Expected type: str")
                    print(f"   Got type: {type(response_obj)}")
                    print(f"   Response object: {response_obj}")
            else:
                tests_failed_count += 1
                print(f"❌ Test 'simple_exact_match': FAILED - No results or unexpected result format.")
                print(f"   Results list: {results}")

        except Exception as e:
            tests_failed_count += 1
            print(f"❌ Test 'simple_exact_match': FAILED - Exception during execution: {e}")
            logger.error("Exception details:", exc_info=True) # Log full traceback

        # --- Report validation status ---
        print(f"\n--- Test Summary ---")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {tests_passed_count}")
        print(f"Failed: {tests_failed_count}")

        if tests_failed_count == 0:
            print("\n✅ VALIDATION COMPLETE - All validation client tests passed.")
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED - Validation client test failed.")
            sys.exit(1)

    # Run the async test function
    asyncio.run(run_test())