# LLM Integration CLI Usage Scenarios

This document outlines realistic usage scenarios for the `llm-integration-cli` tool (`src/pdf_extractor/llm_integration/cli.py`).

## `ask` Command Scenarios

### Scenario 1: Processing a Batch Request from a File

| Field        | Description                                                                                                |
|--------------|------------------------------------------------------------------------------------------------------------|
| **Command**  | `uv run src/pdf_extractor/llm_integration/cli.py ask batch_request.json`                                     |
| **Question** | How do I process multiple LLM tasks defined in a file, potentially with dependencies between them?           |
| **Usage**    | When an agent needs to execute a predefined set of related LLM queries (e.g., summarize text, then extract keywords, then classify sentiment) stored in `batch_request.json`. |
| **Results**  | Executes the batch processing engine, performs LLM calls (respecting dependencies and concurrency limits), validates results, and prints validation reports and the final JSON response to the console. |
| **Outcome**  | The agent gets structured results for all tasks in the batch, allowing for complex, multi-step LLM workflows to be executed via a single command. |

### Scenario 2: Processing a Batch Request from a JSON String

| Field        | Description                                                                                                                               |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Command**  | `uv run src/pdf_extractor/llm_integration/cli.py ask '{"tasks": [{"task_id": "simple_q", "question": "What is 2+2?", "model": "gpt-4o-mini"}]}'` |
| **Question** | How can I quickly run a single, simple LLM task without creating a file?                                                                    |
| **Usage**    | When an agent needs to perform a quick, ad-hoc LLM query or a small batch defined directly on the command line.                             |
| **Results**  | Executes the batch processing engine for the single task provided in the JSON string, performs the LLM call, validates the result, and prints the validation report and JSON response. |
| **Outcome**  | The agent gets a quick answer or result from the LLM for a simple task without the overhead of creating a request file.                     |

