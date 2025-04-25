# ArangoDB Integration: Simple Task Plan

This plan helps you fix and test search features in the ArangoDB system. Follow each step carefully. If you get stuck, read the **Help** section for each task.

---
## Validation Rules
- **MUST MATCH EXPECTED RESULTS**: Every function must produce the exact output specified in the test. A script running without errors is NOT enough. See `VALIDATION_REQUIREMENTS.md`.
- **USE FIXTURES**: Compare results against expected data in `src/test_fixtures/`.
- **DETAILED ERRORS**: If validation fails, print what went wrong (e.g., "Expected X, Got Y").

## Rules to Follow
1. **One Task at a Time**: Finish each task before starting the next.
2. **Test Everything**: Run the test command for each task.
3. **Check Boxes**: Update the checklist in this file by changing `[ ]` to `[x]` for each item. Do this ONLY after the test shows "✅" and matches expected results.
4. **Log Completion**: After checking all boxes, add a line to `task_completion.log` (e.g., "Task 1 completed: All checklist items checked").
5. **Track Failures**: If a test fails, log the attempt in `task_completion.log` (e.g., "Task 1 attempt 1 failed: [error message]").
6. **Use ask-perplexity**: If a task fails twice, use `ask-perplexity` to research the error. Run:
```bash
   ask-perplexity "ArangoDB [specific error message] solution 2025"
```
7. Ask Human: If a task fails three times (or ask-perplexity doesn’t help), stop and log
```bash
echo "Task X failed 3 times: [error details]. Need human help." >> task_completion.log
```
Then STOP and wait for human assistance.
8. *Next Task*: Do not start the next task until the current task’s checklist is checked and logged or escalated to a human.
9. Keep Files Short: Every file must be under 500 lines. Check with wc -l filename.
10. Add File Headers: Every file must start with:
  - Description of the file’s purpose.
  - Links to third-party package docs (e.g., python-arango, litellm).
  - Sample input and expected output.
11. Use Functions: Write functions, not classes, unless you need to maintain state or use Pydantic.
12. Ignore Pylance: Do NOT fix Pylance/type errors until the test matches expected results.
13. Simple Code: Copy the example code if unsure.

---

## Phase 1: Fix Search Features

### Task 0: Project Access
Login to Ubuntu as the user graham with the terminal command:
```ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 -t "cd /home/graham/workspace/experiments/pdf_extractor/ && zsh"```

### Task 1: Fix Imports
**Goal**: Make sure all search files can load their code correctly.

**Steps**:
1. Open these files in a text editor:
   - [x] Fix imports in `src/pdf_extractor/arangodb/search_api/bm25.py`
   - [x] Fix imports in `src/pdf_extractor/arangodb/search_api/semantic.py`
   - [x] Fix imports in `src/pdf_extractor/arangodb/search_api/search_basic.py`
   - [x] Fix imports in `src/pdf_extractor/arangodb/search_api/hybrid.py`
   - [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/keyword.py`
   - [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/glossary.py`
2. Look for lines starting with `import`. If they show errors (e.g., "Module not found"), fix them:
   - Check if the module is in `requirements.txt`.
   - Example: If `import arango` fails, ensure `python-arango` is in `requirements.txt`.
3. Create a new file `src/pdf_extractor/arangodb/test_imports.py`:
   - [x] Created test_imports.py
   - [x] Tested imports work correctly
4. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_imports.py
   ```
5. Check output. It should say "✅ All imports work!".

**Help**:
- If a module is missing, add it to `requirements.txt` and run:
  ```bash
  uv pip install -r requirements.txt
  ```
- If you see "Module not found", check the spelling of the import.

**Checklist**:
- [x] Fixed imports in `bm25.py`
- [x] Fixed imports in `semantic.py`
- [x] Fixed imports in `search_basic.py`
- [x] Fixed imports in `hybrid.py`
- [x] Fixed imports in `keyword.py`
- [x] Fixed imports in `glossary.py`
- [x] Created and tested `test_imports.py`

---

### Task 2: Fix BM25 Search
**Goal**: Make keyword search work using BM25.

**Steps**:
1. Open `src/pdf_extractor/arangodb/search_api/bm25.py`.
2. Look for errors in the `search_bm25` function. Common issues:
   - Missing database connection.
   - Wrong collection name.
   - Wrong view name references.
3. Fix the code. Make sure all references to "mcp_doc_retriever" are changed to "pdf_extractor".
4. Create a test file `src/pdf_extractor/arangodb/test_bm25_search.py`:
   ```python
   from pdf_extractor.arangodb.search_api.bm25 import search_bm25
   from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
   
   # Connect to database
   client = connect_arango()
   db = ensure_database(client)
   
   # Test BM25 search
   results = search_bm25(db, "python json", top_n=5)
   
   # Check if search works
   if results and "results" in results and len(results["results"]) > 0:
       print(f"✅ BM25 search works! Found {results['total']} results")
   else:
       print("❌ BM25 search failed: No results found")
       exit(1)
   ```
5. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_bm25_search.py
   ```
6. Check output. It should say "✅ BM25 search works!".

**Help**:
- If the database connection fails, check `.env` file:
  ```dotenv
  ARANGO_HOST="http://localhost:8529"
  ARANGO_USER="root"
  ARANGO_PASSWORD="openSesame"
  ARANGO_DB="doc_retriever"
  ```
- If you get "VIEW_NAME not defined" error, check that you're using BASE_VIEW_NAME in your code.
- If no results are found, ensure `lessons_learned` collection has data.

**Checklist**:
- [x] Fixed `bm25.py`
- [x] Created `test_bm25_search.py`
- [x] Test passed with "✅ BM25 search works!"

---

### Task 3: Fix Semantic Search
**Goal**: Make meaning-based search work using embeddings.

**Steps**:
1. Open `src/pdf_extractor/arangodb/search_api/semantic.py`.
2. Check for errors in the `search_semantic` function. Common issues:
   - Missing embedding API key.
   - Wrong embedding model name.
   - Incorrect imports.
3. Fix the code. Make sure all references to "mcp_doc_retriever" are changed to "pdf_extractor".
4. Create a test file `src/pdf_extractor/arangodb/test_semantic_search.py`:
   ```python
   from pdf_extractor.arangodb.search_api.semantic import search_semantic
   from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
   
   # Connect to database
   client = connect_arango()
   db = ensure_database(client)
   
   # Test semantic search
   results = search_semantic(db, "python json", top_n=5)
   
   # Check if search works
   if results and "results" in results and len(results["results"]) > 0:
       print(f"✅ Semantic search works! Found {results['total']} results")
   else:
       print("❌ Semantic search failed: No results found")
       exit(1)
   ```
5. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_semantic_search.py
   ```
6. Check output. It should say "✅ Semantic search works!".

**Help**:
- If embedding fails, check `.env` for:
  ```dotenv
  OPENAI_API_KEY="sk-..."
  EMBEDDING_MODEL="text-embedding-3-small"
  ```
- If no results, ensure documents in `lessons_learned` have an `embedding` field.
- If you see syntax errors, check for missing parentheses or quotes.

**Checklist**:
- [x] Fixed `semantic.py`
- [x] Created `test_semantic_search.py`
- [x] Test passed with "✅ Semantic search works!"

---

### Task 4: Fix Hybrid Search
**Goal**: Combine BM25 and semantic search.

**Steps**:
1. Open `src/pdf_extractor/arangodb/search_api/hybrid.py`.
2. Check for errors in the `search_hybrid` function. Common issues:
   - References to BM25 and semantic search functions.
   - Merging logic errors.
3. Fix the code. Make sure all references to "mcp_doc_retriever" are changed to "pdf_extractor".
4. Create a test file `src/pdf_extractor/arangodb/test_hybrid_search.py`:
   ```python
   from pdf_extractor.arangodb.search_api.hybrid import search_hybrid
   from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
   
   # Connect to database
   client = connect_arango()
   db = ensure_database(client)
   
   # Test hybrid search
   results = search_hybrid(db, "python json", top_n=5)
   
   # Check if search works
   if results and "results" in results and len(results["results"]) > 0:
       print(f"✅ Hybrid search works! Found {results['total']} results")
   else:
       print("❌ Hybrid search failed: No results found")
       exit(1)
   ```
5. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_hybrid_search.py
   ```
6. Check output. It should say "✅ Hybrid search works!".

**Help**:
- If no results, ensure BM25 and semantic searches work (Tasks 2 and 3).
- If merge fails, check the logic that combines results from both searches.
- Look for "KeyError" messages which indicate a missing field.

**Checklist**:
- [x] Fixed `hybrid.py`
- [x] Created `test_hybrid_search.py`
- [x] Test passed with "✅ Hybrid search works!"

---

### Task 4.1: Fix Keyword Search
**Goal**: Make keyword search work properly.

**Steps**:
1. Open `src/pdf_extractor/arangodb/search_api/keyword.py`.
2. Fix the issues with the code, including:
   - Hardcoded credentials
   - Hardcoded database and collection names
   - Missing configuration parameters
3. Convert to a proper function with parameters.
4. Create a test file `src/pdf_extractor/arangodb/test_keyword_search.py`.
5. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_keyword_search.py
   ```
6. Check output. It should say "✅ Keyword search works!".

**Checklist**:
- [x] Fixed `keyword.py`
- [x] Created `test_keyword_search.py`
- [x] Test passed with "✅ Keyword search works!"

### Task 4.2: Fix Glossary Search
**Goal**: Make glossary search work properly.

**Steps**:
1. Open `src/pdf_extractor/arangodb/search_api/glossary.py`.
2. Fix the issues with the code, including:
   - Convert to a proper function with parameters
   - Add database connectivity
   - Make the glossary configurable
3. Create a test file `src/pdf_extractor/arangodb/test_glossary_search.py`.
4. Run the test:
   ```bash
   uv run src/pdf_extractor/arangodb/test_glossary_search.py
   ```
5. Check output. It should say "✅ Glossary search works!".

**Checklist**:
- [x] Fixed `glossary.py`
- [x] Created `test_glossary_search.py`
- [x] Test passed with "✅ Glossary search works!"

## Phase 2: Implement Validation

### Task 5: Create Validation Utility
**Goal**: Create a utility for validating search results.

**Steps**:
1. Create folder `src/pdf_extractor/arangodb/validation`:
   ```bash
   mkdir -p src/pdf_extractor/arangodb/validation
   ```
2. Create file `src/pdf_extractor/arangodb/validation/validation_utils.py`:
   ```python
   import os
   import json
   import sys
   from typing import Dict, Any, List, Tuple
   from pathlib import Path
   from loguru import logger
   
   # Path to fixtures
   FIXTURES_DIR = os.path.abspath(
       os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_fixtures")
   )
   
   def ensure_fixtures_dir():
       """Ensure fixtures directory exists."""
       Path(FIXTURES_DIR).mkdir(parents=True, exist_ok=True)
   
   def save_fixture(fixture_name: str, data: Dict) -> str:
       """Save test data to a fixture file."""
       ensure_fixtures_dir()
       fixture_path = os.path.join(FIXTURES_DIR, f"{fixture_name}.json")
       with open(fixture_path, "w") as f:
           json.dump(data, f, indent=2)
       return fixture_path
   
   def load_fixture(fixture_name: str) -> Dict:
       """Load test data from a fixture file."""
       fixture_path = os.path.join(FIXTURES_DIR, f"{fixture_name}.json")
       if not os.path.exists(fixture_path):
           return {}
       with open(fixture_path, "r") as f:
           return json.load(f)
   
   def compare_results(expected: Dict, actual: Dict) -> Tuple[bool, List[Dict]]:
       """Compare expected and actual results."""
       validation_failures = []
       
       # Check total count
       if expected.get("total", 0) != actual.get("total", 0):
           validation_failures.append({
               "field": "total",
               "expected": expected.get("total", 0),
               "actual": actual.get("total", 0)
           })
       
       # Check number of results
       expected_results = expected.get("results", [])
       actual_results = actual.get("results", [])
       if len(expected_results) != len(actual_results):
           validation_failures.append({
               "field": "results_count",
               "expected": len(expected_results),
               "actual": len(actual_results)
           })
       
       # Check first result key fields (if any results exist)
       if expected_results and actual_results:
           expected_keys = set(expected_results[0].keys())
           actual_keys = set(actual_results[0].keys())
           if expected_keys != actual_keys:
               validation_failures.append({
                   "field": "result_fields",
                   "expected": list(expected_keys),
                   "actual": list(actual_keys)
               })
       
       return len(validation_failures) == 0, validation_failures
   
   def report_validation(passed: bool, failures: List[Dict], name: str):
       """Report validation results."""
       if passed:
           logger.info(f"✅ {name} validation passed!")
           return True
       else:
           logger.error(f"❌ {name} validation failed!")
           for failure in failures:
               logger.error(f"  - {failure['field']}: Expected {failure['expected']}, Got {failure['actual']}")
           return False
   ```
3. Run the validation utility to check for errors:
   ```bash
   uv run -c "import pdf_extractor.arangodb.validation.validation_utils; print('✅ Validation utils loaded successfully!')"
   ```

**Help**:
- If you see a module not found error, check import paths.
- If directories don't exist, create them with `mkdir -p`.

**Checklist**:
- [x] Created validation directory
- [x] Created `validation_utils.py`
- [x] Tested loading validation utils

---

### Task 6: Implement BM25 Validation
**Goal**: Create validation for BM25 search.

**Steps**:
1. Create file `src/pdf_extractor/arangodb/validation/validate_bm25.py`:
   ```python
   import sys
   import os
   from loguru import logger
   
   # Add root directory to path
   _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
   if _root not in sys.path:
       sys.path.insert(0, _root)
   
   from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
   from pdf_extractor.arangodb.search_api.bm25 import search_bm25
   from pdf_extractor.arangodb.validation.validation_utils import (
       save_fixture,
       load_fixture,
       compare_results,
       report_validation
   )
   
   def validate_bm25_search():
       """Validate BM25 search functionality."""
       client = connect_arango()
       db = ensure_database(client)
       
       # Execute search
       results = search_bm25(db, "python json", top_n=5)
       
       # Load or create fixture
       fixture = load_fixture("bm25_search_expected")
       if not fixture:
           # First run - save as fixture
           fixture_path = save_fixture("bm25_search_expected", results)
           logger.info(f"Created new fixture: {fixture_path}")
           return True
       
       # Compare against fixture
       passed, failures = compare_results(fixture, results)
       return report_validation(passed, failures, "BM25 search")
   
   if __name__ == "__main__":
       # Configure logging
       logger.remove()
       logger.add(
           sys.stderr,
           level="INFO",
           format="{time:HH:mm:ss} | {level:<7} | {message}"
       )
       
       # Run validation
       if validate_bm25_search():
           sys.exit(0)
       else:
           sys.exit(1)
   ```
2. Run the validation:
   ```bash
   uv run src/pdf_extractor/arangodb/validation/validate_bm25.py
   ```
3. Check output. First run should create fixture, second should validate.

**Help**:
- First run will create a fixture with current results.
- Second run will validate against the fixture.
- If validation fails, check BM25 search function.

**Checklist**:
- [x] Created `validate_bm25.py`
- [x] First run created fixture
- [x] Second run passed validation

---

## Testing Instructions

Test each task one at a time and in order:

```bash
# Task 1: Test imports
uv run src/pdf_extractor/arangodb/test_imports.py

# Task 2: Test BM25 search
uv run src/pdf_extractor/arangodb/test_bm25_search.py

# Task 3: Test semantic search
uv run src/pdf_extractor/arangodb/test_semantic_search.py

# Task 4: Test hybrid search
uv run src/pdf_extractor/arangodb/test_hybrid_search.py

# Task 4.1: Test keyword search
uv run src/pdf_extractor/arangodb/test_keyword_search.py

# Task 4.2: Test glossary search
uv run src/pdf_extractor/arangodb/test_glossary_search.py

# Task 5: No explicit test needed

# Task 6: Validate BM25 search
uv run src/pdf_extractor/arangodb/validation/validate_bm25.py
```

---

## Rules to Follow

1. **Do One Task at a Time**: Finish each task before starting the next.
2. **Test Everything**: Always run the test command for each task.
3. **Check Output**: Only mark a task done if you see "✅".
4. **Use Simple Code**: Copy the example code if you're unsure.
5. **Ask for Help**: If a test fails, read the **Help** section.

---

## Common Problems and Fixes

- **"Module not found"**:
  - Check import paths - should be "pdf_extractor" not "mcp_doc_retriever"
  - Run `uv pip install -r requirements.txt`
- **Database connection failed**:
  - Check `.env` file for correct settings
  - Make sure ArangoDB is running
- **No search results**:
  - Check if collection exists and has data
- **Test fails**:
  - Read the error message carefully
  - Check if you copied the example code correctly
