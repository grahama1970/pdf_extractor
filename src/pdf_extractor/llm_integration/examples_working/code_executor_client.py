# src/pdf_extractor/code_executor_client.py

import requests
import json
from loguru import logger
from typing import Dict, Any, Optional



class CodeExecutorClient:
    """Client for interacting with the Code Executor service."""
    
    def __init__(self, base_url: str = "http://0.0.0.0:8000"):
        """Initialize the client with the service URL."""
        self.base_url = base_url
        logger.info(f"Initialized Code Executor client with base URL: {self.base_url}")
        
    def execute_code(self, 
                    code: str, 
                    language: str = "python", 
                    timeout: Optional[int] = None,
                    max_memory_mb: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute code and return the results.
        
        Args:
            code: The code to execute
            language: Programming language (currently only 'python' supported)
            timeout: Optional timeout in seconds
            max_memory_mb: Optional memory limit in MB
            
        Returns:
            Dictionary with execution results
        """
        payload = {
            "code": code,
            "language": language
        }
        
        if timeout is not None:
            payload["timeout"] = timeout
            
        if max_memory_mb is not None:
            payload["max_memory_mb"] = max_memory_mb
            
        try:
            logger.debug(f"Sending code execution request: {payload}")
            response = requests.post(f"{self.base_url}/execute", json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing code: {str(e)}")
            return {
                "execution_id": "error",
                "status": "error",
                "stdout": "",
                "stderr": f"API request failed: {str(e)}",
                "execution_time": 0,
                "exit_code": -1
            }
    
    def check_health(self) -> bool:
        """Check if the service is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False