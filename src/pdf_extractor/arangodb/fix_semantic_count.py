#!/usr/bin/env python
"""
Script to fix the error in semantic.py related to cursor count not being enabled.
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

def fix_semantic_cursor_count():
    """Update the semantic.py file to enable cursor count"""
    try:
        file_path = "src/pdf_extractor/arangodb/search_api/semantic.py"
        with open(file_path, "r") as f:
            content = f.read()
        
        # Find the line with the cursor execution
        problem_line = "cursor = db.aql.execute(aql, bind_vars=bind_vars, stream=False) # Use stream=False for single result"
        fixed_line = "cursor = db.aql.execute(aql, bind_vars=bind_vars, stream=False, count=True) # Use count=True to enable cursor count"
        
        if problem_line in content:
            content = content.replace(problem_line, fixed_line)
            
            # Write back the fixed content
            with open(file_path, "w") as f:
                f.write(content)
            
            logger.success(f"Successfully fixed the cursor count issue in {file_path}")
            return True
        else:
            logger.warning(f"The problematic line was not found in {file_path}")
            
            # Check if a similar line exists
            similar_lines = [line for line in content.split('\n') if 'cursor = db.aql.execute(' in line]
            if similar_lines:
                logger.info(f"Found similar lines: {similar_lines}")
                
                # Try to find and fix the closest matching line
                for line in similar_lines:
                    if 'stream=False' in line and 'count=True' not in line:
                        new_line = line.replace('stream=False', 'stream=False, count=True')
                        content = content.replace(line, new_line)
                        
                        # Write back the fixed content
                        with open(file_path, "w") as f:
                            f.write(content)
                        
                        logger.success(f"Successfully fixed the cursor count issue in {file_path}")
                        return True
            
            logger.error(f"Could not find a line to fix in {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error fixing semantic.py: {e}")
        return False

if __name__ == "__main__":
    logger.info("Fixing cursor count issue in semantic.py...")
    success = fix_semantic_cursor_count()
    sys.exit(0 if success else 1)
