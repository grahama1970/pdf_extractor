# ArangoDB Integration: Detailed Implementation Plan

## Phase 1: Fix Basic Search Functionality 

### Task 1: Fix Import Issues
- [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/bm25.py`
- [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/semantic.py`
- [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/hybrid.py`
- [ ] Fix imports in `src/pdf_extractor/arangodb/search_api/search_basic.py`
- [ ] Create `test_imports.py` to verify all imports work correctly

### Task 2: Fix and Test BM25 Search
- [ ] Fix `src/pdf_extractor/arangodb/search_api/bm25.py` implementation
- [ ] Create `test_bm25_search.py` with a single test case
- [ ] Document expected inputs/outputs in a comment block
- [ ] Verify search returns expected results

### Task 3: Fix and Test Semantic Search
- [ ] Fix `src/pdf_extractor/arangodb/search_api/semantic.py` implementation
- [ ] Create `test_semantic_search.py` with a single test case
- [ ] Fix vector embedding functionality if broken
- [ ] Verify search returns expected results

### Task 4: Fix and Test Hybrid Search
- [ ] Fix `src/pdf_extractor/arangodb/search_api/hybrid.py` implementation
- [ ] Create `test_hybrid_search.py` with a single test case
- [ ] Ensure it properly combines BM25 and semantic results
- [ ] Verify search returns expected results

## Phase 2: Implement Validation for Search

### Task 5: Create Validation Utility
- [ ] Create `src/pdf_extractor/arangodb/validation/validation_utils.py`
- [ ] Implement field-by-field comparison function
- [ ] Implement detailed error reporting
- [ ] Create fixture management functions

### Task 6: Implement BM25 Validation
- [ ] Create `src/pdf_extractor/arangodb/validation/validate_bm25.py`
- [ ] Create test fixtures with expected results
- [ ] Implement validation against fixtures
- [ ] Add detailed error reporting

### Task 7: Implement Semantic Search Validation
- [ ] Create `src/pdf_extractor/arangodb/validation/validate_semantic.py`
- [ ] Create test fixtures with expected results
- [ ] Implement validation against fixtures
- [ ] Add detailed error reporting

### Task 8: Implement Hybrid Search Validation
- [ ] Create `src/pdf_extractor/arangodb/validation/validate_hybrid.py`
- [ ] Create test fixtures with expected results
- [ ] Implement validation against fixtures
- [ ] Add detailed error reporting

## Phase 3: Lessons Learned Management (After Search Works)

### Task 9: Implement Lessons CRUD Operations
- [ ] Create `src/pdf_extractor/arangodb/lessons_crud.py`
- [ ] Implement create function
- [ ] Implement read function
- [ ] Implement update function
- [ ] Implement delete function
- [ ] Create `test_lessons_crud.py` to verify all operations

### Task 10: Implement Validation for Lessons CRUD
- [ ] Create `src/pdf_extractor/arangodb/validation/validate_lessons_crud.py`
- [ ] Create test fixtures with expected results
- [ ] Implement validation against fixtures
- [ ] Add detailed error reporting

### Task 11: Implement Lessons CLI
- [ ] Create `src/pdf_extractor/arangodb/lessons_cli.py`
- [ ] Implement add command
- [ ] Implement get command
- [ ] Implement list command
- [ ] Implement update command
- [ ] Implement delete command
- [ ] Create `test_lessons_cli.py` to verify all commands

### Task 12: Implement Validation for Lessons CLI
- [ ] Create `src/pdf_extractor/arangodb/validation/validate_lessons_cli.py`
- [ ] Create test fixtures with expected results
- [ ] Implement validation against fixtures
- [ ] Add detailed error reporting

## Testing Instructions

Test each file individually:

```bash
# Test import fixes
uv run src/pdf_extractor/arangodb/test_imports.py

# Test BM25 search
uv run src/pdf_extractor/arangodb/test_bm25_search.py

# Test semantic search
uv run src/pdf_extractor/arangodb/test_semantic_search.py

# Test hybrid search
uv run src/pdf_extractor/arangodb/test_hybrid_search.py

# Run validations
uv run src/pdf_extractor/arangodb/validation/validate_bm25.py
uv run src/pdf_extractor/arangodb/validation/validate_semantic.py
uv run src/pdf_extractor/arangodb/validation/validate_hybrid.py

# After search functionality is complete:
uv run src/pdf_extractor/arangodb/test_lessons_crud.py
uv run src/pdf_extractor/arangodb/validation/validate_lessons_crud.py
uv run src/pdf_extractor/arangodb/test_lessons_cli.py
uv run src/pdf_extractor/arangodb/validation/validate_lessons_cli.py
```

## Implementation Guidelines

1. **One File at a Time**: Complete each file before moving to the next
2. **Functional First**: Focus on making functionality work before improving code
3. **Simple is Better**: Use the simplest approach that works
4. **95/5 Rule**: Use 95% existing functionality with 5% customization
5. **Always Validate**: Verify each implementation with proper validation
6. **Error Details**: Provide detailed error reporting in validation
7. **Check Each Box**: Only mark a task complete when validation passes
