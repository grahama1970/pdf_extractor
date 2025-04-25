# src/pdf_extractor/llm_integration/examples_working/code_generation_example.py

import asyncio
from pdf_extractor.llm_integration.examples_working.code_generation import generate_and_execute_code

# Configure logging
from loguru import logger

async def main():
    prompt = "Create a function to calculate the Fibonacci sequence up to n terms"
    logger.info(f"Running code generation with prompt: {prompt}")
    
    # Generate and execute code
    result = await generate_and_execute_code(prompt, timeout=60, max_memory_mb=2048)
    
    # Print results
    logger.info(f"Result: {result}")
    print("Code Generation and Execution Result:")
    print(f"Success: {result['success']}")
    print(f"Generated Code:\n{result['code']}")
    if result['execution_result']:
        print("Execution Result:")
        print(f"Status: {result['execution_result']['status']}")
        print(f"Output: {result['execution_result']['stdout']}")
        print(f"Errors: {result['execution_result']['stderr']}")
        print(f"Execution Time: {result['execution_result']['execution_time']}")
        print(f"Exit Code: {result['execution_result']['exit_code']}")
    else:
        print("No execution result available")

if __name__ == "__main__":
    asyncio.run(main())