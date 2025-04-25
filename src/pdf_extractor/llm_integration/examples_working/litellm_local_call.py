from typing import Dict, List
from litellm import acompletion #type: ignore
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import asyncio

load_dotenv()

# Configuration

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
)
async def get_llm_response(config: dict) -> str:
    """
    Get response from LLM model using configuration dictionary.
    
    Args:
        config: Dictionary containing API configuration and prompt
    
    Returns:
        str: Model's response
    """
    api_params: Dict[str, object] = {
        "model": config['model'],
        "messages": [
            {"role": "user", "content": config['prompt']}
        ],
    }

    if config['local']:
        api_params["api_base"] = config['local_api_base']
        api_params["api_key"] = config['local_api_key']
    
    try:
        response = await acompletion(**api_params) # type: ignore
        result = response.choices[0].message.content # type: ignore
        return result.strip() if isinstance(result, str) else result #type: ignore

    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return ""

async def main() -> None:
    config = {
        'local_api_base': os.getenv('LOCAL_API_BASE', "http://localhost:4000/v1/"),
        'default_local_model': os.getenv('DEFAULT_LOCAL_MODEL', "openai/codellama-7b"),
        'local_api_key': os.getenv('LOCAL_API_KEY', "anything"),
        'prompt': "What is the capital of France?",
        'model': "openai/codellama-7b",
        'local': True
    }
    
    response = await get_llm_response(config)
    print(response)

if __name__ == "__main__":
   asyncio.run(main())
