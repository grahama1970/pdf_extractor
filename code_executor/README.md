# Code Executor Service 🚀

The **Code Executor Service** is a secure, containerized API for running Python code in isolated environments. Built with FastAPI and Docker, it ensures safe execution of user-submitted scripts with resource limits, network isolation, and automatic cleanup. Perfect for testing code snippets, running data analysis, or integrating into larger applications! 🐍

## Why Use This Service? 🌟

This service is designed for developers and applications needing to execute Python code securely and reliably. Here's why it's awesome:
- **Security**: Runs code in isolated Docker containers with restricted permissions (no network, limited filesystem access) 🔒.
- **Flexibility**: Supports popular Python libraries like `numpy`, `pandas`, `matplotlib`, `scipy`, and `sklearn` 📊.
- **Resource Control**: Limits CPU, memory, and execution time to prevent abuse ⏱️.
- **Scalability**: Built with FastAPI for high-performance API handling 🚀.
- **Ease of Use**: Simple JSON-based API for submitting and retrieving results 📨.

Use cases include:
- Testing Python scripts in a sandbox.
- Running data processing tasks with `pandas` or machine learning with `sklearn`.
- Integrating code execution into educational platforms or IDEs.

## How It Works 🛠️

The service uses a robust architecture:
- **FastAPI Server**: Handles API requests (`/execute`, `/health`) with JSON payloads.
- **Python Subprocess**: Executes code in a separate process with resource limits (memory, CPU, processes) using `subprocess.Popen`.
- **Docker Container**: Runs the server in a `python:3.11-slim` image with pre-installed modules and a read-only filesystem (except `/tmp`).
- **Code Sanitization**: Uses `ast` to block dangerous operations (e.g., `os.system`, `subprocess.run`).
- **Dynamic Imports**: Only loads required modules (e.g., `pandas`) when needed, optimizing performance.
- **Temporary Files**: Stores code and outputs in `/tmp`, cleaned up after execution.

The flow:
1. Client submits code via `POST /execute`.
2. Server validates the code for safety.
3. Code runs in a subprocess with resource limits.
4. Output (`stdout`, `stderr`) is captured and returned as JSON.
5. Temporary files are deleted, ensuring a clean state.

## Setup Instructions 🏗️

Follow these steps to get the service running locally.

### Prerequisites
- **Docker**: Installed and running 🐳.
- **Git**: To clone the repository (optional).
- **curl** or **Postman**: For testing API endpoints.

### Directory Structure
```plaintext
pdf_extractor/
├── code_executor/
│   ├── app/
│   │   ├── main.py
│   │   ├── entry_point.sh
│   ├── requirements.txt
├── Dockerfile
├── README.md

Steps

Clone the Repository (or create the structure manually):
git clone <repository-url>
cd pdf_extractor


Ensure Required Files:

Dockerfile: Defines the container (based on python:3.11-slim).
code_executor/app/main.py: FastAPI server logic.
code_executor/app/entry_point.sh: Launches the server.
code_executor/requirements.txt: Lists dependencies (fastapi, uvicorn, pydantic, psutil).


Build the Docker Image:
docker build -t code_executor --no-cache .


Run the Container:
docker run --name code_executor -d -p 8000:8000 --memory="4g" --cpus="2" code_executor


Maps port 8000 for API access.
Allocates 4GB memory and 2 CPUs for stability.


Verify the Service:
docker logs code_executor

Look for:

Uvicorn running on http://0.0.0.0:8000.
System resources: CPUs=..., Memory=....


Test the Health Endpoint:
curl http://localhost:8000/health

Expected: {"status": "healthy"}.


API Endpoints 📡
POST /execute
Execute Python code in a secure environment.
Request Body (JSON):
{
  "code": "print('Hello, World!')",
  "language": "python",
  "execution_id": "optional-custom-id",
  "timeout": 60,
  "max_memory_mb": 2048
}


code: Python code to execute (required).
language: Must be "python" (required).
execution_id: Optional custom ID (defaults to UUID).
timeout: Execution time limit in seconds (defaults to 300, max 600).
max_memory_mb: Memory limit in MB (defaults to 4096, max 8192).

Response (JSON):
{
  "execution_id": "3e225e35-6945-4641-8aca-b49cdf41399c",
  "status": "completed",
  "stdout": "Hello, World!\n",
  "stderr": "",
  "execution_time": 1.578629493713379,
  "exit_code": 0
}


status: completed, timeout, or error.
stdout: Captured standard output (up to 10MB).
stderr: Captured error output.
execution_time: Time taken in seconds.
exit_code: Subprocess exit code (0 for success).

Example:
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello, World!\")", "language": "python"}' | jq

GET /health
Check if the service is running.
Response (JSON):
{
  "status": "healthy"
}

Example:
curl http://localhost:8000/health

Usage Examples 📚
Simple Script
Run a basic print statement:
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Test\")", "language": "python"}' | jq

Output:
{
  "execution_id": "<uuid>",
  "status": "completed",
  "stdout": "Test\n",
  "stderr": "",
  "execution_time": <float>,
  "exit_code": 0
}

Complex Script with Imports
Run a pandas script:
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import pandas as pd; df = pd.DataFrame({\"A\": [1, 2, 3]}); print(df.mean())", "language": "python", "timeout": 60, "max_memory_mb": 2048}' | jq

Output:
{
  "execution_id": "<uuid>",
  "status": "completed",
  "stdout": "A    2.0\ndtype: float64\n",
  "stderr": "",
  "execution_time": <float>,
  "exit_code": 0
}


Note: Modules (numpy, pandas, etc.) are pre-installed and resolve automatically. Set BYPASS_SETUP=false to pre-import modules:docker exec code_executor bash -c "export BYPASS_SETUP=false"
docker restart code_executor



Handling Import Errors
Try an uninstalled module:
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import tensorflow; print(\"Hi\")", "language": "python"}' | jq

Output:
{
  "execution_id": "<uuid>",
  "status": "completed",
  "stdout": "",
  "stderr": "ImportError: No module named 'tensorflow'\nEnsure required modules are installed and in ALLOWED_MODULES.",
  "execution_time": <float>,
  "exit_code": 1
}

Unsafe Code
Try malicious code:
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "import os; os.system(\"ls\")", "language": "python"}' | jq

Output:
{
  "detail": "Unsafe code detected"
}

Configuration ⚙️
The service is configured via environment variables in the Dockerfile:

EXECUTION_TIMEOUT=300: Default timeout (seconds).
MAX_MEMORY_MB=4096: Default memory limit (MB).
MAX_OUTPUT_SIZE=10485760: Max output size (10MB).
ALLOWED_MODULES=numpy,pandas,matplotlib,scipy,sklearn: Pre-installed modules.
BYPASS_SETUP=true: Skips pre-importing modules (set to false for faster module loading).

Override variables at runtime:
docker run --name code_executor -d -p 8000:8000 \
  -e BYPASS_SETUP=false \
  --memory="4g" --cpus="2" code_executor

Security Considerations 🔐

Code Sanitization: Blocks dangerous operations (os.system, subprocess.run) using ast.
Resource Limits: Enforces memory, CPU, file, and process limits via resource.setrlimit.
Docker Isolation: Runs with no network access and limited capabilities (add --cap-drop=ALL --security-opt=no-new-privileges for production).
Temporary Files: Stored in /tmp and deleted after execution.
Read-Only Filesystem: Only /tmp is writable.

For production, enhance security:
docker run --name code_executor -d -p 8000:8000 \
  --memory="4g" --cpus="2" \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --network=none \
  code_executor

Troubleshooting 🐞

Container Fails to Start:
Check logs: docker logs code_executor.
Ensure port 8000 is free.


Execution Hangs:
Increase timeout or max_memory_mb in the request.
Set BYPASS_SETUP=false to pre-import modules.
Check module availability: docker exec code_executor pip list.


Import Errors:
Verify the module is in ALLOWED_MODULES and installed.
Reinstall: docker exec code_executor pip install --force-reinstall numpy.


Resource Issues:
Monitor: docker stats code_executor.
Check /tmp: docker exec code_executor df -h /tmp.



Contributing 🤝
Contributions are welcome! To add features or fix bugs:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

License 📜
MIT License. See LICENSE for details.
Happy coding! 🎉```
