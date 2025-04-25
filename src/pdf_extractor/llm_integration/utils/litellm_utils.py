import os
from typing import Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv
from litellm import model_cost as litellm_model_cost # Rename import to avoid conflict
import litellm

load_dotenv()

# Add missing model costs if not present in litellm's default
# Note: Costs can change, these are examples based on common pricing.
# Check official provider pricing for accuracy.
_EXTRA_MODEL_COSTS = {
    "openai/gpt-4o-mini": {
        "input_cost_per_token": 0.00000015, # $0.15 / 1M tokens
        "output_cost_per_token": 0.0000006, # $0.60 / 1M tokens
    },
    # Add other custom/missing models here if needed
    "vertex_ai/gemini-2.5-pro-exp-03-25": { # Add vertex entry, assuming free for now
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
     "xai/grok-3-mini-beta": { # Ensure grok is present if needed
        "input_cost_per_token": 0.0000003,  # Example cost $0.30 / 1M
        "output_cost_per_token": 0.0000005, # Example cost $0.50 / 1M
    }
}

# Merge extra costs into litellm's dictionary (be careful not to overwrite existing)
for model, cost_data in _EXTRA_MODEL_COSTS.items():
    if model not in litellm_model_cost:
        litellm_model_cost[model] = cost_data
        logger.debug(f"Added missing cost data for model: {model}")
    # Optionally update existing if needed, but might conflict with library updates
    # else:
    #     litellm_model_cost[model].update(cost_data)


def model_cost_per_million_tokens(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns the input, output, and total cost per million tokens for a specific model.
    Uses litellm.model_cost and includes fallbacks for missing models.
    Logs errors if the model or cost data is missing/invalid.
    """
    # Use the potentially updated litellm_model_cost dictionary
    info = litellm_model_cost.get(model_name) 
    if not info:
        logger.error(f"Model '{model_name}' not found in model_cost dictionary.")
        available_models = ', '.join(list(litellm_model_cost.keys())[:10]) # Show more examples
        logger.info(f"Available models in cost map: {available_models} ...")
        return None

    try:
        # Check for specific cost keys first
        input_cost_token = info.get('input_cost_per_token')
        output_cost_token = info.get('output_cost_per_token')

        # Fallback for different naming conventions (e.g., per 1k tokens)
        if input_cost_token is None:
             input_cost_1k = info.get('input_cost_per_1k_tokens')
             if input_cost_1k is not None:
                  input_cost_token = float(input_cost_1k) / 1000
        if output_cost_token is None:
             output_cost_1k = info.get('output_cost_per_1k_tokens')
             if output_cost_1k is not None:
                  output_cost_token = float(output_cost_1k) / 1000
        
        # Final check if costs are still missing
        if input_cost_token is None or output_cost_token is None:
            # Check if it's explicitly free (like our added vertex entry)
            if info.get('input_cost_per_token') == 0.0 and info.get('output_cost_per_token') == 0.0:
                 input_cost_token = 0.0
                 output_cost_token = 0.0
            else:
                 raise ValueError(f"Missing cost per token data (input: {input_cost_token}, output: {output_cost_token}). Info: {info}")

        input_cost = float(input_cost_token) * 1_000_000
        output_cost = float(output_cost_token) * 1_000_000
        total_cost = input_cost + output_cost

        # Log costs only if they are non-zero, reduce noise for free models
        if total_cost > 0:
            logger.debug(f"Costs for Model: {model_name}")
            logger.debug(f"  Input cost per million tokens:  ${input_cost:.6f}") # More precision
            logger.debug(f"  Output cost per million tokens: ${output_cost:.6f}")
            logger.debug(f"  Total cost per million tokens:  ${total_cost:.6f}")
        else:
             logger.debug(f"Model {model_name} is free or has zero cost listed.")


        return {
            "model_name": model_name, 
            "input_cost": input_cost, 
            "output_cost": output_cost, 
            "total_cost": total_cost
        }

    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating costs for model '{model_name}': {e}. Cost info: {info}")
        return None

if __name__ == "__main__":
    # Example usage: replace with your desired model name
    test_models = ["openai/gpt-4o-mini", "vertex_ai/gemini-2.5-pro-exp-03-25", "xai/grok-3-mini-beta", "nonexistent-model"]
    for model_name in test_models:
        print(f"\n--- Testing cost for: {model_name} ---")
        result = model_cost_per_million_tokens(model_name)
        if result:
            print(f"Result: {result}")
        else:
            print("Could not retrieve cost.")
