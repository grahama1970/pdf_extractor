import requests
import json

def run_code(code):
    # url = "http://192.168.86.49:8000/execute"
    url = "http://localhost:8000/execute"
    payload = {
        "code": code,
        "language": "python"
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Simple test
result = run_code('print("Hello, World!")')
print(json.dumps(result, indent=2))

# Test with numpy
numpy_code = """
import numpy as np
data = np.random.rand(5, 5)
print("Random matrix:")
print(data)
print("\\nMatrix sum:", np.sum(data))
"""
result = run_code(numpy_code)
print(json.dumps(result, indent=2))