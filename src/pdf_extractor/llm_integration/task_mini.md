# LLM Integration Refinement Task Plan

**Goal:** Integrate validation reporting into the CLI, enhance corpus loading flexibility, and ensure the full LLM integration pipeline functions correctly.

**Current Status:**
- `engine.py` updated to handle validation strategies and results (string, file, directory corpus).
- `retry_llm_call.py` updated to perform validation and return status.
- Validation utilities (`validation_utils/`) are in place.
- `cli.py` modified and verified to call `report_validation_results`.
- `main.py` standalone execution block updated and verified for end-to-end testing, including enhanced corpus loading.

**Tasks:**

- [x] **1. Create Task Plan:** Write this plan to `src/pdf_extractor/llm_integration/task_mini.md`.
- [x] **2. Modify `cli.py`:** Integrate `report_validation_results` from `validation_utils.reporting` to display validation results.
- [x] **3. Verify `cli.py` Reporting:** Run a relevant `cli.py` command and confirm the report format.
- [x] **4. Verify Full Pipeline (`main.py` - Basic):** Execute `main.py`'s updated `__main__` block to ensure end-to-end functionality.
- [x] **5. Enhance Corpus Loading (`engine.py`):**
    - [x] 5a. Add dependencies: `PyMuPDF`, `html2text`, `bleach`, `reportlab`.
    - [x] 5b. Modify `engine.py` to handle `corpus_type` (`string`, `file`, `directory`) in `validation_options`.
    - [x] 5c. Implement loading/text extraction for `.pdf`, `.json`, `.md`, `.html` files.
    - [x] 5d. Implement directory scanning with `file_patterns` and `recursive` options.
    - [x] 5e. Implement error handling (skip bad files, log warnings, fail task if corpus required but none loaded).
- [x] **6. Update `main.py` Validation:** Modify the `__main__` block in `main.py` to include test cases for the new file/directory corpus loading.
- [x] **7. Verify Enhanced Pipeline (`main.py`):** Execute `main.py`'s updated `__main__` block to ensure enhanced corpus loading works correctly.
- [ ] **8. Final Review & Completion:** Confirm all steps are done and report completion.