# PDF Extractor - Directory Structure Guide

## Project Organization

The project follows a clean, organized structure with these key directories:



## Important Guidelines

1. **Add new code to appropriate module directories**:
   - Core functionality goes in 
   - Tests go in 
   - Experimental code goes in 

2. **Don't add files to the src/ root directory**:
   - This clutters the project
   - Makes imports and module structure confusing
   - Always place files in the appropriate subdirectory

3. **Keep the module structure clean**:
   - Use  to expose the right functions/classes
   - Follow the established import patterns
   - Maintain proper directory hierarchy

4. **Use the temp directory for temporary files**:
   - Place debug scripts, temporary tests, etc. in 
   - Don't commit these files to the repository

## Module Structure

The core PDF extractor module has these primary components:

1. **Extraction Components**:
   - : Extract content using Marker
   - : Enhanced table detection and merging
   - : Table extraction with Camelot integration
   - : Qwen-VL-2B integration for visual elements

2. **API & Interface**:
   - : FastAPI server implementation
   - : Command-line interface

3. **Core Conversion**:
   - : Core PDF-to-JSON conversion logic

Always maintain this structure when adding new components.
