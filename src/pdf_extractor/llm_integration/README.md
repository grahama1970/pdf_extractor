# MCP LiteLLM Service 💬🚀

## Overview 🌟

`mcp-litellm-service` is a Dockerized FastAPI application designed to act as a Model Context Protocol (MCP) server for agents like Roo Code or Claude. It provides a robust interface for executing batch requests against various Large Language Models (LLMs) supported by the excellent LiteLLM library.

This service intelligently handles concurrent execution for independent questions and sequential execution for questions with dependencies. It incorporates sophisticated retry mechanisms for both API call resilience (using Tenacity) and response content validation/correction (using a custom retry wrapper). It leverages Redis for caching LLM responses, improving performance and reducing costs.

This project is intended to be built and potentially maintained using an agentic workflow, specifically following the Roomodes framework described below.

## ✨ Features

*   ✅ **Batch Processing:** Accepts multiple questions in a single API request.
*   ✅ **Concurrent & Sequential Execution:** Automatically handles independent questions concurrently and dependent questions sequentially based on input order and `method` flag.
*   ✅ **Multi-LLM Support:** Leverages LiteLLM to interact with numerous LLM providers (OpenAI, Anthropic, Gemini, Cohere, local models via Ollama, etc.).
*   ✅ **Configurable Validation:** Supports per-question validation strategies (defaulting to Pydantic model validation if a `response_model` is specified).
*   ✅ **Dual Retry Mechanisms:**
    *   Handles transient API errors using Tenacity within `litellm_call`.
    *   Handles response content validation errors using a higher-level retry loop (`retry_llm_call`) that modifies prompts for correction.
*   ✅ **Redis Caching:** Integrates with LiteLLM's caching backend using Redis for optimized performance.
*   ✅ **Structured I/O:** Uses Pydantic models for strict validation of API requests and guarantees structured JSON responses.
*   ✅ **MCP Compatible:** Designed to be easily integrated as an MCP server in environments like Roo Code or Claude MCP.
*   ✅ **Dockerized:** Packaged as a Docker container using `docker-compose` for easy deployment and dependency management (includes Redis service).
*   ✅ **Standard Packaging:** Uses `pyproject.toml` and `uv` for dependency management.

## 🏗️ Runtime Architecture Diagram

This diagram illustrates the flow of a request through the service at runtime:

```mermaid
graph TD
    subgraph "External Agent (e.g., Roo Code)"
        Agent -- "HTTP POST /ask\n(BatchRequest JSON)" --> FAPI
    end

    subgraph "Docker Container: mcp-litellm-service"
        FAPI(🌐 FastAPI - main.py) -- Parse Request (models.py) --> Engine
        Engine(⚙️ Asyncio Engine - engine.py) -- Manages Tasks --> RetryWrap
        RetryWrap(🔄 Retry Wrapper - retry_llm_call.py) -- "Call with Validation Strategy" --> LLMCall
        LLMCall(📞 LiteLLM Call - litellm_call.py) -- Use Tenacity Retries --> LiteLLM
        LiteLLM(💡 LiteLLM Core) -- API Call --> LLMAPIs[☁️ External LLM APIs]
        LiteLLM -- Check/Store --> Redis[(💾 Redis Cache)]

        subgraph "Validation & Retry Loop (Retry Wrapper)"
            direction TB
            RetryWrap -- "Validate Response\n(Pydantic/Custom)" --> V{Validation Check}
            V -- Validation OK --> Engine
            V -- "Validation Failed\n(Max Retries Not Reached)" --> ModifyPrompt[📝 Modify Prompt]
            ModifyPrompt -- Call Again --> LLMCall
        end

        Engine -- "Collect Results\n(ResultItem - models.py)" --> FAPI
        FAPI -- "Format Response\n(BatchResponse JSON)" --> AgentResp(HTTP Response)
    end

    AgentResp --> Agent

    %% Styling
    classDef default fill:#f9f,stroke:#333,stroke-width:2px
    classDef agent fill:#ccf,stroke:#333
    classDef fastapi fill:#9cf,stroke:#333
    classDef engine fill:#9fc,stroke:#333
    classDef retry fill:#ff9,stroke:#333
    classDef llm fill:#f99,stroke:#333
    classDef external fill:#ccc,stroke:#333
    classDef data fill:#eee,stroke:#666,stroke-dasharray: 5 5

    class Agent,AgentResp agent;
    class FAPI fastapi;
    class Engine engine;
    class RetryWrap,V,ModifyPrompt retry;
    class LLMCall,LiteLLM llm;
    class Redis,LLMAPIs external;
Use code with caution.
Markdown
🛠️ Technology Stack
Language: Python 3.10+

Web Framework: FastAPI

LLM Abstraction: LiteLLM

Data Validation: Pydantic

Caching: Redis

Concurrency: Asyncio

Retry Logic: Tenacity

Containerization: Docker, Docker Compose

Dependency Management: uv, pyproject.toml

🤖 Roomodes Workflow (Project Construction)
This project is designed to be built and maintained using the Roomodes framework, leveraging specialized AI agents for different tasks. The primary flow for building features or fixing bugs involves:

📝 Planner: Reads the detailed task plan document (like the one used to generate this README). It identifies the next incomplete task ([ ]) and delegates it.

🪃 Boomerang Mode: Receives the task from Planner. Analyzes it and delegates it to the most appropriate specialist Coder agent. It orchestrates any necessary back-and-forth (e.g., for clarification) and reports the final outcome back to Planner.

🧑‍💻 Coder Agents (Intern/Junior/Senior): Receive specific coding tasks from Boomerang Mode.

Crucially: Before coding, they are responsible for downloading relevant documentation for required libraries (e.g., FastAPI, LiteLLM) into the repo_docs/ directory using tools like git sparse-checkout.

They implement the code, adhering to standards:

Using uv add / uv sync for dependencies via pyproject.toml.

Adding standard header comments to each file (description, doc links, sample I/O).

Keeping file sizes under 500 lines.

Implementing inline usage examples (if __name__ == "__main__": or test scripts) instead of formal unit tests.

Mandatory: Grepping the downloaded documentation in repo_docs/ when encountering errors or needing to understand library usage.

Report completion/results back to Boomerang Mode.

(Other roles like Hacker, Librarian, Researcher are defined in the Roomodes setup but typically not directly involved in building this specific service).

📁 Project Structure
mcp-litellm-service/
├── .git/
├── .gitignore
├── .env.example        # Example environment variables
├── .venv/              # Virtual environment (managed by uv)
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Docker build instructions
├── pyproject.toml      # Project metadata and dependencies (for uv)
├── uv.lock             # Locked dependency file
├── README.md           # This file
├── repo_docs/          # Downloaded third-party documentation (managed by Coder Agent)
│   └── ...
├── scripts/            # Utility and testing scripts
│   ├── smoke_test_litellm.py
│   └── test_api.py
└── src/
    └── mcp_litellm/      # Main application source code
        ├── __init__.py
        ├── engine.py         # Core async execution logic
        ├── initialize_litellm_cache.py # Redis cache setup
        ├── litellm_call.py   # Wrapper around LiteLLM API call with Tenacity
        ├── main.py           # FastAPI application entrypoint
        ├── models.py         # Pydantic models for API and data structures
        ├── parser.py         # Result substitution logic
        ├── retry_llm_call.py # Content validation retry wrapper
        └── llm_base.py       # (If needed for type hinting/structure)
Use code with caution.
⚙️ Setup & Installation
Prerequisites:

Docker Desktop (or Docker Engine)

Docker Compose (usually included with Docker Desktop)

Git

Python 3.10+ (for running uv locally if needed)

uv (Install via pip install uv or preferred method)

Steps:

Clone the Repository:

git clone <repository-url>
cd mcp-litellm-service
Use code with caution.
Bash
Configure Environment Variables 🔑:

Copy the example environment file:

cp .env.example .env
Use code with caution.
Bash
Edit the .env file and add your necessary API keys for the LLM providers you intend to use (e.g., OpenAI, Perplexity, Anthropic). LiteLLM uses standard environment variable names.

# Example .env content
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Add other keys as needed by LiteLLM
Use code with caution.
Dotenv
docker-compose.yml is configured to load variables from this .env file.

(Optional) Local Development Setup:

Create and activate a virtual environment: uv venv

Install dependencies: uv sync

🚀 Running the Service
Build the Docker Images:

docker compose build
Use code with caution.
Bash
Start the Services (FastAPI + Redis):

docker compose up -d
Use code with caution.
Bash
This will start the mcp-litellm-service container (exposing port 8000 by default) and the redis container.

Verify Service is Running:

Check container logs:

docker compose logs mcp_litellm_service
docker compose logs redis
Use code with caution.
Bash
You should see Uvicorn startup messages for the FastAPI service.

Stopping the Services:

docker compose down
Use code with caution.
Bash
💻 API Usage
The service exposes a single primary endpoint:

Endpoint: POST /ask

Request Body: A JSON object conforming to the BatchRequest model.

questions: (List[QuestionItem], required) - A list where each item represents a question to be processed.

index: (int, required) - A unique identifier for the question within the batch.

question: (str, required) - The text of the question/prompt for the LLM. Can contain placeholders like {Q<index>_result} for sequential dependencies.

method: (str, optional, default: 'concurrent') - Either 'concurrent' or 'sequential'. Sequential tasks depend implicitly on the result of the preceding sequential task with a lower index in the original request list.

model: (str, optional, default: 'openai/gpt-4o-mini') - The LiteLLM model identifier (e.g., gpt-4o-mini, claude-3-opus-20240229, gemini/gemini-1.5-pro-latest, ollama/mistral).

validation_strategy: (str, optional, default: 'pydantic') - Strategy for response validation. 'pydantic' enables validation/retries if response_model is also set. Future strategies could be added (e.g., 'citation_check').

response_model: (str, optional, default: None) - The name of a Pydantic model (defined in models.py) to validate the LLM response against. Required for 'pydantic' validation strategy.

temperature: (float, optional, default: 0.2) - LLM temperature.

max_tokens: (int, optional, default: 1000) - Max tokens for the LLM response.

api_base: (str, optional, default: None) - Override LiteLLM base URL if needed.

(Other LiteLLM parameters can be added to QuestionItem as needed)

Example Request JSON:

{
  "questions": [
    {
      "index": 1,
      "question": "What is the approximate average temperature in Dublin, Ireland during January in Celsius?",
      "method": "sequential",
      "model": "gemini/gemini-1.5-flash"
    },
    {
      "index": 2,
      "question": "Based on the previous answer ({Q1_result}), what type of clothing (e.g., light jacket, heavy coat, layers) would you recommend packing for a trip there in January?",
      "method": "sequential",
      "model": "openai/gpt-4o-mini"
    },
    {
      "index": 3,
      "question": "What are the primary colors?",
      "method": "concurrent",
      "model": "openai/gpt-4o-mini"
    }
  ]
}
Use code with caution.
Json
Response Body: A JSON object conforming to the BatchResponse model.

results: (List[ResultItem]) - A list containing the outcome for each question in the batch, typically sorted by index.

index: (int) - The index of the original question.

status: (str) - 'success' or 'error'.

result: (Any, optional) - The successful response from the LLM. This could be a string or a parsed Pydantic model object if response_model was used. null if status is 'error'.

error_message: (str, optional) - Details about the error if status is 'error'. null otherwise.

confidence: (float, optional) - Confidence score, if provided by the underlying LLM API (e.g., Perplexity). null otherwise.

retry_count: (int, optional) - Number of content validation retries performed by retry_llm_call for this question. Defaults to 0 if no validation retries occurred.

Example Response JSON:

{
  "results": [
    {
      "index": 1,
      "status": "success",
      "result": "The approximate average temperature in Dublin, Ireland during January is around 5°C.",
      "error_message": null,
      "confidence": null,
      "retry_count": 0
    },
    {
      "index": 2,
      "status": "success",
      "result": "Given an average temperature of around 5°C, it's recommended to pack warm clothing, including layers, a heavy coat, scarf, gloves, and a hat for a trip to Dublin in January. Waterproof outerwear is also advisable due to potential rain.",
      "error_message": null,
      "confidence": null,
      "retry_count": 0
    },
    {
      "index": 3,
      "status": "success",
      "result": "The primary colors are typically considered to be red, yellow, and blue.",
      "error_message": null,
      "confidence": null,
      "retry_count": 0
    }
  ]
}
Use code with caution.
Json
Example curl Usage:

curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{
  "questions": [
    { "index": 1, "question": "What is 1+1?", "model": "openai/gpt-4o-mini" },
    { "index": 2, "question": "What color is the sky?", "model": "openai/gpt-4o-mini" }
  ]
}'
Use code with caution.
Bash
Example Python requests Usage: (See scripts/test_api.py)

import requests
import json

api_url = "http://localhost:8000/ask"
payload = {
  "questions": [
    { "index": 1, "question": "What is the capital of France?", "model": "openai/gpt-4o-mini" }
  ]
}

response = requests.post(api_url, json=payload)

if response.status_code == 200:
  print(json.dumps(response.json(), indent=2))
else:
  print(f"Error: {response.status_code}")
  print(response.text)
Use code with caution.
Python
🤔 Key Concepts Explained
Concurrency vs. Sequential: Questions marked concurrent (or default) are run in parallel using asyncio.gather. Questions marked sequential wait for the result of the immediately preceding sequential question in the original request list before executing. Result substitution ({Q<index>_result}) allows using previous answers in subsequent prompts. The agent sending the request is responsible for ordering sequential questions correctly.

Validation Strategies: The validation_strategy field (default 'pydantic') determines how/if the LLM response content is validated. Currently, 'pydantic' requires a corresponding response_model field naming a Pydantic model; if validation fails, the retry_llm_call wrapper attempts correction. Other strategies could be added later. If no specific validation is needed, the retry wrapper might still run but without specific content checks.

Retry Mechanisms:

API Call Retries (Tenacity): The inner litellm_call uses Tenacity to automatically retry the litellm.acompletion call if transient network or server errors occur during the API request itself.

Content Validation Retries: The outer retry_llm_call wrapper invokes litellm_call, then validates the response content based on the validation_strategy. If validation fails, it adds a correction message to the prompt and calls litellm_call again, up to a maximum number of tries. The retry_count in the response reflects these content retries.

Caching: LiteLLM's built-in caching is enabled and configured to use the linked Redis service. Subsequent identical requests (model, messages, parameters) should hit the cache, reducing latency and cost.

🔌 MCP Integration
To use this service within an MCP environment like Roo Code:

Ensure the Docker container is built and preferably pushed to a registry accessible by your MCP host, or ensure Docker Compose can build/run it locally on the host.

Add a configuration block to your mcp_settings.json file (global or project-specific).

Example mcp_settings.json entry:

{
  "mcpServers": {
    "mcp-litellm-batch": { // Choose a unique name for the server
      "command": "docker", // Or http if MCP calls API directly
      "args": [
        // Example using docker run:
        "run",
        "--rm", // Remove container after exit
        "-p", "8000:8000", // Expose the port
        // Pass API keys securely from the MCP host's environment
        "-e", "OPENAI_API_KEY=${env.OPENAI_API_KEY}",
        "-e", "PERPLEXITY_API_KEY=${env.PERPLEXITY_API_KEY}",
        // Add other necessary keys...
        // Use --network if needed to connect to other MCP services
        // "--network", "your_mcp_network",
        "your-dockerhub-username/mcp-litellm-service:latest" // Replace with your image path
        // OR use docker compose commands if preferred
      ],
      "env": {
        // Example if MCP uses direct HTTP: Define base URL
        // "API_ENDPOINT": "http://localhost:8000/ask" // Or service name if in same Docker network
      },
      "disabled": false, // Set to false to enable
      "alwaysAllow": [
        "litellm_batch_ask" // Define the tool name Roo will use to call this MCP
      ]
    }
    // ... other MCP servers
  }
}
Use code with caution.
Json
Tool Definition: You would then need to define the litellm_batch_ask tool within Roo's capabilities, specifying how it should format the input JSON for this MCP server's /ask endpoint based on the agent's request.

🧪 Testing
This project prioritizes inline usage examples and integration testing over formal unit tests (pytest).

Core functionality of individual modules can often be tested by running the Python file directly (if an if __name__ == "__main__": block is provided).

The scripts/smoke_test_litellm.py provides a basic check of LiteLLM interaction.

The primary integration test is scripts/test_api.py, which sends requests to the running Docker service via the FastAPI endpoint. Use this script to verify end-to-end functionality.

📚 Documentation Standards
repo_docs/: Coder agents are expected to download relevant third-party library documentation into this directory before starting work on a task. They must reference (grep) this documentation when implementing features or debugging.

File Headers: Every .py file in src/mcp_litellm/ must include a standard header comment block detailing its purpose, links to relevant documentation (in repo_docs/ or official URLs), sample input, and sample output for its primary function/class.

File Size: Python source files should ideally remain under 500 lines to maintain readability and focus.

