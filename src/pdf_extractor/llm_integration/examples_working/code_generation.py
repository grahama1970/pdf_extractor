# src/pdf_extractor/llm_integration/examples_working/code_generation.py

import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from pdf_extractor.llm_integration.examples_working.code_executor_client import CodeExecutorClient
from pdf_extractor.llm_integration.examples_working.litellm_local_call_batch_validation import get_llm_response

async def generate_and_execute_code(prompt: str, timeout: int = 60, max_memory_mb: int = 2048) -> Dict[str, Any]:
    """
    Generate code using LLM and execute it.
    
    Args:
        prompt: The prompt for code generation
        timeout: Code execution timeout in seconds
        max_memory_mb: Memory limit for code execution in MB
        
    Returns:
        Dictionary with code generation and execution results
    """
    # Step 1: Add code generation instructions to the prompt
    code_prompt = f"""
Write Python code to solve this problem:

{prompt}

Provide only Python code without explanation. Ensure the code is complete, ready to run, and includes a print statement to display the output (e.g., call the function with a sample input).
"""
    
    # Step 2: Generate code using LLM
    logger.debug(f"Sending LLM prompt: {code_prompt}")
    llm_response = await get_llm_response({
        "model": "codellama-7b",
        "prompt": code_prompt,
        "local": True
    })
    
    if not llm_response or not llm_response[0][1]:
        logger.error("Failed to generate code: No response from LLM")
        return {
            "success": False,
            "error": "Failed to generate code",
            "code": None,
            "execution_result": None
        }
    
    generated_code = llm_response[0][1].strip()
    logger.info(f"Generated code:\n{generated_code}")
    
    # Step 3: Execute the generated code
    client = CodeExecutorClient()
    execution_result = client.execute_code(
        code=generated_code,
        timeout=timeout,
        max_memory_mb=max_memory_mb
    )
    logger.info(f"Execution result: {execution_result}")
    
    # Step 4: Return results
    return {
        "success": execution_result["exit_code"] == 0,
        "code": generated_code,
        "execution_result": execution_result
    }