# PDF Extractor Project - Optimized Context Document

## Current Status

We've implemented an improved table merger module that enhances PDF extraction by detecting and merging multi-page tables. The key files are:

- : Core implementation with three merge strategies
- : Updated to use the improved merger
- : Basic tests for the functionality
- : Documentation of the module

## Directory Structure Status

The project directory has become cluttered with many temporary files and test scripts. The core files have been properly organized, but there's still cleanup needed:



## Key Module Features

### Improved Table Merger
- Three merge strategies:
  - : Only merges tables with high similarity
  - : Merges tables with more relaxed requirements
  - : Disables merging functionality
- Enhanced header detection with partial matching
- Safe merging with proper validation

### Integration with Table Extractor
- Added  parameter to TableExtractor class
- Automatic merging during table processing
- Detailed logging of merge operations

## Next Tasks (In Priority Order)

1. **Directory Cleanup**:
   - Move temporary test files to .temp directory
   - Consolidate duplicate files
   - Ensure proper module structure

2. **Comprehensive Testing**:
   - Test with real PDFs from src/input directory
   - Validate results with all three merge strategies
   - Create proper test fixtures

3. **Documentation Enhancement**:
   - Add usage examples with code snippets
   - Document integration approaches
   - Add performance considerations

4. **Integration with Pipeline**:
   - Ensure compatibility with marker_processor
   - Test end-to-end PDF processing

## Core Development Standards

- Follow 95/5 rule: 95% package functionality, 5% customization
- Validate all outputs against expected results
- Use  for package management, not pip
- Handle errors gracefully with proper logging
