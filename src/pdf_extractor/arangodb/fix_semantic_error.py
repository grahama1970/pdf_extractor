#!/usr/bin/env python
"""
Script to fix the error in semantic.py related to the AQLQueryExecuteError object
that doesn't have an http_status_code attribute.
"""
import sys
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:HH:mm:ss} | {level:<7} | {message}"
)

def fix_semantic_error_handler():
    """Update the error handling in semantic.py to not rely on http_status_code"""
    try:
        file_path = "src/pdf_extractor/arangodb/search_api/semantic.py"
        with open(file_path, "r") as f:
            content = f.read()
        
        # Find and replace the problematic line
        problem_line = "f\"Semantic AQL Error (ID: {search_uuid}): Code={e.error_code}, Msg='{e.error_message}'. HTTP Status: {e.http_status_code}\\nQuery:\\n{aql}\","
        fixed_line = "f\"Semantic AQL Error (ID: {search_uuid}): Code={e.error_code}, Msg='{e.error_message}'\\nQuery:\\n{aql}\","
        
        if problem_line in content:
            content = content.replace(problem_line, fixed_line)
            
            # Write back the fixed content
            with open(file_path, "w") as f:
                f.write(content)
            
            logger.success(f"Successfully fixed the error in {file_path}")
            return True
        else:
            logger.warning(f"The problematic line was not found in {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing semantic.py: {e}")
        return False

if __name__ == "__main__":
    logger.info("Fixing error in semantic.py...")
    success = fix_semantic_error_handler()
    sys.exit(0 if success else 1)
