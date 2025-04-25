# LiteLLM Module Integration: Task List

> This document outlines tasks for integrating and ensuring the litellm module meets pdf_extractor project standards

## Overview
The litellm module provides a robust framework for querying large language models with batch processing, concurrent/sequential execution, and result management. This integration will enable pdf_extractor to leverage models like Perplexity AI and Gemini 2.5 Pro (with 1M context length) for processing PDF content.

## Project Information
- **Project**: pdf_extractor
- **Path**: `/home/graham/workspace/experiments/pdf_extractor`
- **Python Version**: 3.11.12
- **Package Manager**: uv (not pip)

## Priority Tasks

### Phase 1: Code Quality and Standards Alignment

- [x] **Task 1.1: Package Structure Verification**
  - [ ] Verify all `__init__.py` files are properly set up
  - [ ] Ensure imports are working correctly
  - [ ] Fix any circular import issues
  - [ ] Confirm module can be imported from pdf_extractor
  - **Validation**: Import the module from a test script in the project root

- [ ] **Task 1.2: Code Standards Alignment** 
  - [ ] Conform `litellm_call.py` to CODING_PRINCIPLES.md (.claude/project_docs/CODING_PRINCIPLES.md) (95/5 rule)
  - [ ] Align `retry_llm_call.py` with error handling practices
  - [ ] Update `main.py` to match logging standards
  - [ ] Ensure all files have proper docstrings with expected formats
  - **Validation**: Run static analysis tools on the code

- [ ] **Task 1.3: Validation Standards Implementation**
  - [ ] Add validation functions to each module's `__main__` block
  - [ ] Create expected outputs for each key function
  - [ ] Implement proper assertion checks
  - [ ] Ensure validation reports detailed error information
  - **Validation**: Each module should run standalone with validation according to .claude/project_docs/VALIDATION_REQUIREMENTS.md

### Phase 2: Configuration and Environment Setup

- [ ] **Task 2.1: Environment Configuration**
  - [ ] Create a proper `.env.example` file with required API keys:
    - OpenAI
    - Anthropic (Claude)
    - Google (Gemini)
    - Perplexity
  - [ ] Add ArangoDB connection parameters
  - [ ] Document required environment variables
  - **Validation**: Test environment loading with sample values

- [ ] **Task 2.2: ArangoDB Schema Setup**
  - [ ] Create script for setting up required collections
  - [ ] Add vector indices for `lesson_embedding` field
  - [ ] Set up validation for database operations
  - **Validation**: Run connection test script that validates schema

- [ ] **Task 2.3: Dependency Management**
  - [ ] Create a requirements.txt file for the module
  - [ ] Update pyproject.toml with required dependencies
  - [ ] Set version constraints for packages
  - [ ] Ensure compatibility with existing project dependencies
  - **Validation**: Test installation with uv

### Phase 3: Component Testing and Debugging

- [ ] **Task 3.1: Core Function Testing**
  - [ ] Create test for `litellm_call.py` with LLM connection
  - [ ] Test each model provider separately
  - [ ] Validate response formats
  - [ ] Fix any authentication or connection issues
  - **Validation**: Successfully query each LLM provider

- [ ] **Task 3.2: Batch Processing Testing**
  - [ ] Test concurrent question processing
  - [ ] Test sequential question processing with dependencies
  - [ ] Validate placeholder substitution works correctly
  - [ ] Ensure error handling correctly captures and reports issues
  - **Validation**: Run test batch with mixed concurrent/sequential tasks

- [ ] **Task 3.3: Retry and Error Handling**
  - [ ] Test API failure scenarios
  - [ ] Verify retry logic works with backoff
  - [ ] Test validation failure handling
  - [ ] Ensure errors are properly logged and reported
  - **Validation**: Run test with forced errors, verify recovery

- [ ] **Task 3.4: Fix Known Bugs from lessons_learned.json**
  - [ ] Address attribute access patterns (`item.method` vs `item.mode`)
  - [ ] Fix message formatting for LLM calls
  - [ ] Correct dictionary structure handling
  - [ ] Fix error propagation issues
  - **Validation**: Run tests that previously failed

### Phase 4: PDF Integration

- [ ] **Task 4.1: Create PDF Content Processor**
  - [ ] Develop function to extract key information from PDF JSON
  - [ ] Create question generator based on PDF content
  - [ ] Format questions with proper LLM prompts
  - [ ] Implement result association with PDF sections
  - **Validation**: Process sample PDF outputs into LLM queries

- [ ] **Task 4.2: Implement Integration Module**
  - [ ] Create main interface for pdf_extractor to litellm
  - [ ] Develop result handler for LLM responses
  - [ ] Implement batch function for processing multiple PDFs
  - [ ] Add logging for integration steps
  - **Validation**: End-to-end test with sample PDF

- [ ] **Task 4.3: Long Context Processing**
  - [ ] Test and optimize for large PDF content
  - [ ] Implement chunking strategy for models with context limits
  - [ ] Add support for 1M token context with Gemini 2.5 Pro
  - [ ] Create test cases with very large PDFs
  - **Validation**: Process PDF larger than 100 pages successfully

### Phase 5: Documentation and Usage Examples

- [ ] **Task 5.1: Update Documentation**
  - [ ] Add documentation for integration module
  - [ ] Create examples of common usage patterns
  - [ ] Document environment setup requirements
  - [ ] Add troubleshooting guide
  - **Validation**: Review documentation for clarity and completeness

- [ ] **Task 5.2: Create Usage Examples**
  - [ ] Simple example for single PDF processing
  - [ ] Batch processing example for multiple PDFs
  - [ ] Example with custom validation logic
  - [ ] Advanced usage with ArangoDB storage
  - **Validation**: Test each example script

## Implementation Examples

### Task 1.3: Validation Example for `litellm_call.py`

```python
def validate_litellm_call():
    """Validate litellm_call functionality with expected outputs."""
    
    # Define expected outputs
    EXPECTED_RESULTS = {
        "content_present": True,
        "content_type": str,
        "min_content_length": 10,
        "usage_present": True
    }
    
    # Run test call
    config = {
        "llm_config": {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Respond with 'Test successful'"}],
            "temperature": 0.0,
            "max_tokens": 50
        }
    }
    
    try:
        response = asyncio.run(litellm_call(config))
        
        # Validate response structure
        validation_results = {}
        
        # Check if content is present and correct type
        content = None
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
                
        validation_results["content_present"] = content is not None
        validation_results["content_type"] = type(content) == str
        validation_results["min_content_length"] = len(content) >= EXPECTED_RESULTS["min_content_length"] if content else False
        validation_results["usage_present"] = hasattr(response, 'usage')
        
        # Check validation results
        validation_passed = all(validation_results.values())
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - litellm_call returned expected response structure")
            return True
        else:
            print("❌ VALIDATION FAILED - litellm_call response doesn't match expected format")
            print("Validation details:")
            for key, result in validation_results.items():
                expected = EXPECTED_RESULTS.get(key, "Not specified")
                print(f"  - {key}: Expected: {expected}, Got: {result}")
            return False
            
    except Exception as e:
        print(f"❌ VALIDATION FAILED - litellm_call raised exception: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    # Run validation
    if validate_litellm_call():
        sys.exit(0)
    else:
        sys.exit(1)
```

### Task 4.1: PDF Content Processor Example

```python
def generate_questions_from_pdf_content(pdf_json, question_types=None):
    """
    Generate LLM questions based on PDF content.
    
    Args:
        pdf_json: JSON output from pdf_extractor
        question_types: List of question types to generate
            (summary, analysis, extraction, etc.)
            
    Returns:
        BatchRequest object with tasks
    """
    if question_types is None:
        question_types = ["summary", "key_points"]
        
    tasks = []
    task_id = 1
    
    # Extract metadata
    metadata = pdf_json.get("metadata", {})
    filename = metadata.get("filename", "unknown.pdf")
    page_count = metadata.get("page_count", 0)
    
    # Get all content as a single string for summary questions
    all_text = ""
    for section in pdf_json.get("sections", []):
        all_text += section.get("text", "") + "\n\n"
        
    # Create summary task
    if "summary" in question_types:
        summary_task = TaskItem(
            task_id=f"summary_{task_id}",
            dependencies=[],
            question=f"Summarize the following document '{filename}' ({page_count} pages) in 3-5 paragraphs:\n\n{all_text[:50000]}",
            model="gemini/gemini-1.5-pro",
            max_tokens=1000
        )
        tasks.append(summary_task)
        task_id += 1
        
    # Create key points task
    if "key_points" in question_types:
        keypoints_task = TaskItem(
            task_id=f"keypoints_{task_id}",
            dependencies=[],
            question=f"Extract the 5-7 most important points from this document '{filename}':\n\n{all_text[:50000]}",
            model="gemini/gemini-1.5-pro",
            max_tokens=1000
        )
        tasks.append(keypoints_task)
        task_id += 1
    
    # Create BatchRequest
    batch_request = BatchRequest(
        tasks=tasks
    )
    
    return batch_request

# Validation
def validate_question_generation():
    """Validate question generation from PDF content."""
    # Define test PDF JSON
    test_pdf = {
        "metadata": {
            "filename": "test.pdf",
            "page_count": 5
        },
        "sections": [
            {"text": "This is a test document with sample content."},
            {"text": "It contains multiple sections for testing."}
        ]
    }
    
    # Define expected outputs
    EXPECTED_RESULTS = {
        "task_count": 2,
        "contains_summary": True,
        "contains_keypoints": True,
        "model_correct": True
    }
    
    # Generate questions
    batch_request = generate_questions_from_pdf_content(test_pdf)
    
    # Validate results
    validation_results = {}
    validation_results["task_count"] = len(batch_request.tasks) == 2
    validation_results["contains_summary"] = any("summary" in task.task_id for task in batch_request.tasks)
    validation_results["contains_keypoints"] = any("keypoints" in task.task_id for task in batch_request.tasks)
    validation_results["model_correct"] = all(task.model == "gemini/gemini-1.5-pro" for task in batch_request.tasks)
    
    # Check validation results
    validation_passed = all(validation_results.values())
    
    # Report validation status
    if validation_passed:
        print("✅ VALIDATION COMPLETE - Question generation works as expected")
        return True
    else:
        print("❌ VALIDATION FAILED - Question generation doesn't match expected output")
        print("Validation details:")
        for key, result in validation_results.items():
            expected = EXPECTED_RESULTS.get(key, "Not specified")
            print(f"  - {key}: Expected: {expected}, Got: {result}")
        return False

if __name__ == "__main__":
    # Run validation
    if validate_question_generation():
        sys.exit(0)
    else:
        sys.exit(1)
```

## Task Tracking Instructions

1. Update this document as you complete tasks:
   - `[ ]` → `[x]` for completed items
   - `[ ]` → `[F]` for failed/obsolete items with a note

2. Use comments to document progress and issues:
   ```
   - [x] Task 1.1: Updated package structure
     Comment: Fixed circular import issues between engine.py and models.py
   ```

3. Add new tasks as needed with proper validation criteria

4. Run validation after completing each task

## Relevant Standards References

- Refer to VALIDATION_REQUIREMENTS.md for validation formats
- Follow CODE_DETAILS.md for code structure requirements
- Adhere to CODING_PRINCIPLES.md for architecture decisions
- Include comprehensive validation in every module