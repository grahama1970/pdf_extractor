# PDF Extractor Project Structure

## Key Directories

- src/pdf_extractor/ - MAIN MODULE DIRECTORY (add new code here)
- src/pdf_extractor/test/ - Test files
- src/pdf_extractor/experiments/ - Experimental code
- docs/ - Documentation
- output/ - Generated files
- .temp/ - Temporary files

## Important Guidelines

1. DO NOT add files directly to src/ - this clutters the project
2. Always place files in the appropriate subdirectory
3. Place temporary files in .temp/
4. Keep module structure clean with proper imports
5. Follow existing patterns for new components

## Core Components

- table_extractor.py - Table extraction with merge strategy integration
- improved_table_merger.py - Multi-page table merging with three strategies
- marker_processor.py - Content extraction using Marker
- pdf_to_json_converter.py - Core PDF to JSON conversion
- api.py - FastAPI server interface
- cli.py - Command-line interface
