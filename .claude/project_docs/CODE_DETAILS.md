# CODE DETAILS
> Comprehensive code standards for PDF extraction

## Execution Standards
- Always use uv for package management; never pip
- Run Python scripts with: `uv run script.py`
- For environment variables, always use the `env` command:
  ```sh
  env VAR_NAME="value" uv run command
  ```

## Validation Requirements
**Every PDF extractor function must validate results:**

```python
# 1. Define expected outputs
EXPECTED_RESULTS = {
  "expected_sections": 2,
  "expected_tables": 1,
  "expected_properties": {"extraction_accuracy": 95.0}
}

# 2. Compare actual results
assert len(extracted_sections) == EXPECTED_RESULTS["expected_sections"]
assert len(extracted_tables) == EXPECTED_RESULTS["expected_tables"]
assert extraction_accuracy >= EXPECTED_RESULTS["expected_properties"]["extraction_accuracy"]

# 3. Report validation status
if validation_passed:
  print("✅ VALIDATION COMPLETE - All PDF extraction results match expected values")
  sys.exit(0)
else:
  print("❌ VALIDATION FAILED - PDF extraction results don't match expected values") 
  print(f"Expected: {expected}, Got: {actual}")
  sys.exit(1)
```

## Module Structure Requirements
- **Documentation**: Every extractor module must include purpose and sample PDFs
- **Size**: No module should exceed 500 lines
- **Validation**: Every module must test against sample PDFs in src/input

## Error Handling for PDFs
1. Handle malformed PDFs gracefully
2. Follow systematic debugging approach:
   - Review PDF structure
   - Test with simpler PDFs
   - Analyze extraction patterns
   - Implement robust fallbacks
