# validate_all.py
import sys
import os
import subprocess
from loguru import logger

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"
    )
    
    logger.info("Running validation for all search API functions")
    
    # Validation scripts to run
    validation_scripts = [
        "validate_bm25.py",
        "validate_semantic.py",
        "validate_hybrid.py"
    ]
    
    # Run each validation script
    results = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script in validation_scripts:
        script_path = os.path.join(current_dir, script)
        logger.info(f"Running {script}...")
        
        try:
            # Run the script as a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True
            )
            
            # Check return code
            if result.returncode == 0:
                logger.success(f"{script} validation passed!")
                results[script] = True
            else:
                logger.error(f"{script} validation failed with code {result.returncode}")
                logger.error(f"Output: {result.stderr}")
                results[script] = False
                
        except Exception as e:
            logger.error(f"Error running {script}: {str(e)}")
            results[script] = False
    
    # Report overall results
    logger.info("=== Validation Results ===")
    for script, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} - {script}")
    
    # Check if all validations passed
    if all(results.values()):
        logger.success("✅ VALIDATION COMPLETE - All search API validations passed!")
        sys.exit(0)
    else:
        failed_scripts = [script for script, success in results.items() if not success]
        logger.error(f"❌ VALIDATION FAILED - The following validations failed: {', '.join(failed_scripts)}")
        sys.exit(1)
