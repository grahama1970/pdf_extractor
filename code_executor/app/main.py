import os
import uuid
import time
import tempfile
import subprocess
import resource
import logging
import select
import psutil
import ast
from typing import Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secure Code Execution Service")

# Configuration
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", 300))
MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", 4096))
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")
ALLOWED_MODULES = os.getenv("ALLOWED_MODULES", "numpy,pandas,matplotlib,scipy,sklearn").split(",")
MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_COMMANDS = ["python"]
BYPASS_SETUP = os.getenv("BYPASS_SETUP", "true").lower() == "true"

# Log system resources at startup
logger.info(f"System resources: CPUs={psutil.cpu_count()}, Memory={psutil.virtual_memory().total / (1024*1024)}MB")
try:
    disk = psutil.disk_usage(TEMP_DIR)
    logger.info(f"Disk space at {TEMP_DIR}: Total={disk.total / (1024*1024)}MB, Free={disk.free / (1024*1024)}MB")
except Exception as e:
    logger.error(f"Failed to check disk space: {e}")

class CodeSubmission(BaseModel):
    code: str
    language: str = "python"
    execution_id: Optional[str] = None
    timeout: Optional[int] = None
    max_memory_mb: Optional[int] = None

class ExecutionResult(BaseModel):
    execution_id: str
    status: str
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int

@contextmanager
def temporary_file(suffix: str, content: str, dir: str = TEMP_DIR):
    """Create and manage a temporary file, ensuring cleanup."""
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, mode='w', dir=dir, delete=False)
    try:
        temp_file.write(content)
        temp_file.close()
        logger.debug(f"Created temporary file: {temp_file.name}")
        yield temp_file.name
    finally:
        try:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                logger.debug(f"Deleted temporary file: {temp_file.name}")
        except Exception as e:
            logger.error(f"Failed to delete {temp_file.name}: {e}")

def set_resource_limits(max_memory_mb: int, timeout: int):
    """Set resource limits for the subprocess."""
    max_memory_bytes = max_memory_mb * 1024 * 1024
    logger.debug(f"Setting resource limits: Memory={max_memory_mb}MB, CPU={timeout}s, Files=128, Processes=100")
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))  # Virtual memory
    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))  # CPU time
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))  # Open files
    resource.setrlimit(resource.RLIMIT_NPROC, (100, 100))  # Max processes

def read_pipe(pipe, max_size: int) -> str:
    """Read from a pipe incrementally, respecting max_size."""
    output = []
    total_size = 0
    while True:
        r, _, _ = select.select([pipe], [], [], 0.1)
        if not r:
            break
        data = pipe.read(1024)
        if not data:
            break
        total_size += len(data)
        if total_size > max_size:
            output.append(data[:max_size - total_size])
            break
        output.append(data)
    return "".join(output)

def is_code_safe(code: str) -> bool:
    """Check if code contains dangerous operations."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["system", "exec", "eval", "open"]:
                    return False
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if any(name.name in ["os", "subprocess", "sys"] for name in node.names):
                    return False
        return True
    except SyntaxError:
        return False

def get_required_modules(code: str) -> set:
    """Detect required modules from code."""
    try:
        tree = ast.parse(code)
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if name.name in ALLOWED_MODULES:
                        modules.add(name.name)
        return modules
    except SyntaxError:
        return set()

def execute_code(code: str, execution_id: str, timeout: int, max_memory_mb: int) -> ExecutionResult:
    """Execute code in a controlled subprocess."""
    start_time = time.time()
    logger.info(f"Starting execution for ID: {execution_id}")
    logger.debug(f"Submitted code (truncated): {code[:100]}{'...' if len(code) > 100 else ''}")

    # Log environment info
    try:
        python_version = subprocess.check_output(["python", "--version"]).decode().strip()
        logger.debug(f"Python version: {python_version}")
        for module in ALLOWED_MODULES:
            try:
                result = subprocess.check_output(["python", "-c", f"import {module}; print('{module} OK')"], 
                                              stderr=subprocess.STDOUT).decode().strip()
                logger.debug(f"Module check: {result}")
            except Exception as e:
                logger.warning(f"Module {module} import failed: {e}")
    except Exception as e:
        logger.error(f"Failed to get Python version: {e}")

    # Create setup file
    required_modules = get_required_modules(code) if not BYPASS_SETUP else set()
    setup_content = "# Import allowed modules\n"
    if not BYPASS_SETUP:
        for module in required_modules:
            setup_content += f"try:\n    import {module}\nexcept ImportError as e:\n    print(f'Failed to import {module}: {{e}}')\n"

    stdout, stderr = "", ""
    exit_code = -1
    status = "error"

    # Write files
    with temporary_file(suffix='.py', content=setup_content) as setup_path:
        with temporary_file(suffix='.py', content=code) as code_path:
            try:
                cmd = ["python", code_path] if BYPASS_SETUP else \
                      ["python", "-c", f"exec(open('{setup_path}').read()); exec(open('{code_path}').read())"]
                logger.info(f"Executing command: {cmd}")

                if cmd[0] not in ALLOWED_COMMANDS:
                    raise ValueError(f"Invalid command: {cmd[0]}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=lambda: set_resource_limits(max_memory_mb, timeout),
                    text=True
                )
                logger.info(f"Subprocess started with PID: {process.pid}")

                # Read output incrementally
                start_time_comm = time.time()
                while process.poll() is None:
                    if time.time() - start_time_comm > timeout:
                        logger.warning(f"Timeout after {timeout} seconds")
                        process.kill()
                        stdout = read_pipe(process.stdout, MAX_OUTPUT_SIZE)
                        stderr = read_pipe(process.stderr, MAX_OUTPUT_SIZE) or \
                                 f"Execution timed out after {timeout} seconds"
                        exit_code = -9
                        status = "timeout"
                        break
                    stdout += read_pipe(process.stdout, MAX_OUTPUT_SIZE - len(stdout))
                    stderr += read_pipe(process.stderr, MAX_OUTPUT_SIZE - len(stderr))
                    logger.debug(f"Partial stdout: {stdout[:100]}{'...' if len(stdout) > 100 else ''}")
                    logger.debug(f"Partial stderr: {stderr[:100]}{'...' if len(stderr) > 100 else ''}")

                if process.poll() is not None and status != "timeout":
                    stdout += read_pipe(process.stdout, MAX_OUTPUT_SIZE - len(stdout))
                    stderr += read_pipe(process.stderr, MAX_OUTPUT_SIZE - len(stderr))
                    exit_code = process.returncode
                    status = "completed"
                    logger.info(f"Subprocess completed with exit code: {exit_code}")

            except subprocess.SubprocessError as e:
                logger.error(f"Subprocess error: {e}")
                stdout = ""
                stderr = f"Subprocess error: {str(e)}"
                if "ImportError" in str(e):
                    stderr += "\nEnsure required modules are installed and in ALLOWED_MODULES."
                exit_code = -1
                status = "error"
            except Exception as e:
                logger.error(f"Execution error: {e}")
                stdout = ""
                stderr = f"Execution error: {str(e)}"
                exit_code = -1
                status = "error"

    execution_time = time.time() - start_time
    logger.info(f"Execution time: {execution_time}s")

    return ExecutionResult(
        execution_id=execution_id,
        status=status,
        stdout=stdout,
        stderr=stderr,
        execution_time=execution_time,
        exit_code=exit_code
    )

@app.post("/execute", response_model=ExecutionResult)
async def run_code(submission: CodeSubmission):
    """API endpoint to execute code."""
    if submission.language != "python":
        raise HTTPException(status_code=400, detail=f"Language {submission.language} not supported")
    if not is_code_safe(submission.code):
        raise HTTPException(status_code=400, detail="Unsafe code detected")

    execution_id = submission.execution_id or str(uuid.uuid4())
    timeout = min(submission.timeout or EXECUTION_TIMEOUT, 600)  # Cap at 10 minutes
    max_memory_mb = min(submission.max_memory_mb or MAX_MEMORY_MB, 8192)  # Cap at 8GB

    logger.info(f"Received execution request: ID={execution_id}, Timeout={timeout}s, Memory={max_memory_mb}MB")

    result = execute_code(submission.code, execution_id, timeout, max_memory_mb)
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
