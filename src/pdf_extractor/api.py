"""
FastAPI endpoints for PDF-to-JSON conversion with Label Studio integration.

This module provides a RESTful API to upload PDF files, process them using the
PDF extraction pipeline, and return structured JSON output. It integrates with
the `pdf_converter` module to handle conversion tasks and Label Studio for
human-in-the-loop validation.

Dependencies:
- fastapi: For building the API.
- pydantic: For request validation.
- uvicorn: For running the server.
- loguru: For logging.
- python-multipart: For file uploads.
- requests: For Label Studio API communication.

Usage:
    Run the server:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```

    Upload a PDF:
    ```bash
    curl -X POST "http://localhost:8000/convert" \
         -F "file=@sample.pdf" \
         -F "repo_link=https://github.com/example/repo"
    ```

    Create Label Studio tasks:
    ```bash
    curl -X POST "http://localhost:8000/label-studio/create-tasks/sample" \
         -F "project_id=1"
    ```

    Check status:
    ```bash
    curl http://localhost:8000/status
    ```
"""

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger
import uvicorn
import requests
from sse_starlette.sse import EventSourceResponse
import asyncio

from pdf_extractor.config import (
    DEFAULT_OUTPUT_DIR, 
    DEFAULT_CORRECTIONS_DIR, 
    DEFAULT_UPLOADS_DIR,
    LABEL_STUDIO_URL,
    LABEL_STUDIO_TOKEN
)
from pdf_extractor.pdf_to_json_converter import convert_pdf_to_json
from .utils import fix_sys_path

# Initialize FastAPI app
app = FastAPI(
    title="PDF to JSON Converter API",
    description="API for converting PDF files to structured JSON using Marker, Camelot, and Qwen-VL with Label Studio integration.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ConversionResponse(BaseModel):
    """Response model for conversion endpoint."""
    status: str
    message: str
    data: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    message: str

class TableInfo(BaseModel):
    """Model for table data."""
    id: str
    page: int
    caption: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any]
    bbox: Optional[List[float]] = None

class ValidationFeedback(BaseModel):
    """Model for validation feedback from Label Studio."""
    task_id: str
    pdf_id: str
    table_id: str
    validation_status: str
    table_data: Optional[Dict] = None
    table_bbox: Optional[List[float]] = None
    merge_target: Optional[str] = None
    comment: Optional[str] = None
    extraction_quality: Optional[int] = None

class ProjectResponse(BaseModel):
    """Response model for Label Studio project creation."""
    project_id: int
    message: str

class TasksResponse(BaseModel):
    """Response model for Label Studio task creation."""
    pdf_id: str
    project_id: int
    task_count: int
    message: str

class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    pdf_id: str
    table_id: str
    status: str
    message: str

class CorrectionResponse(BaseModel):
    """Response model for correction application."""
    pdf_id: str
    status: str
    message: str

class CorrectionStatusResponse(BaseModel):
    """Response model for correction status."""
    pdf_id: str
    status: str
    message: str

# Helper functions for Label Studio integration
def ensure_directories_exist():
    """Ensure all required directories exist."""
    for directory in [DEFAULT_OUTPUT_DIR, DEFAULT_CORRECTIONS_DIR, DEFAULT_UPLOADS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_label_studio_project(project_name: str) -> int:
    """Create a new Label Studio project for PDF table validation."""
    try:
        # Read labeling configuration from file
        config_path = Path(__file__).parent / "labeling_config.xml"
        if not config_path.exists():
            config_path = Path("labeling_config.xml")  # Fallback to root directory
        
        with open(config_path, "r") as f:
            labeling_config = f.read()
        
        # Create project in Label Studio
        response = requests.post(
            f"{LABEL_STUDIO_URL}/projects/",
            headers={
                "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "title": project_name,
                "label_config": labeling_config,
            }
        )
        
        if response.status_code == 201:
            project_id = response.json()["id"]
            logger.info(f"Created Label Studio project: {project_id}")
            return project_id
        else:
            logger.error(f"Failed to create Label Studio project: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to create Label Studio project")
    except Exception as e:
        logger.exception("Error creating Label Studio project")
        raise HTTPException(status_code=500, detail=str(e))

def create_label_studio_tasks(pdf_id: str, project_id: int, extracted_tables: List[TableInfo]) -> int:
    """Create tasks in Label Studio for human validation of extracted tables."""
    try:
        tasks = []
        
        for table in extracted_tables:
            # Prepare table data in the expected format for Label Studio
            task_data = {
                "pdf_id": pdf_id,
                "page": table.page,
                "pdf_page_url": f"/data/upload/uploads/{pdf_id}.pdf#page={table.page}",
                "table_id": table.id,
                "source": table.metadata.get("extraction_method", "unknown"),
                "table_data": {
                    "headers": table.headers,
                    "rows": table.rows
                }
            }
            
            if table.bbox:
                task_data["bbox"] = table.bbox
            
            # Create task in Label Studio
            tasks.append({
                "data": task_data,
                "meta": {
                    "pdf_id": pdf_id,
                    "table_id": table.id,
                    "page": table.page
                }
            })
        
        # Batch create tasks
        if tasks:
            response = requests.post(
                f"{LABEL_STUDIO_URL}/projects/{project_id}/import",
                headers={
                    "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
                    "Content-Type": "application/json"
                },
                json=tasks
            )
            
            if response.status_code == 201:
                logger.info(f"Created {len(tasks)} Label Studio tasks for PDF {pdf_id}")
                
                # Save tasks to corrections directory for reference
                corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{pdf_id}_tasks.json"
                with open(corrections_file, "w") as f:
                    json.dump(tasks, f, indent=2)
                
                return len(tasks)
            else:
                logger.error(f"Failed to create Label Studio tasks: {response.text}")
                raise HTTPException(status_code=500, detail="Failed to create Label Studio tasks")
        
        return 0
    except Exception as e:
        logger.exception("Error creating Label Studio tasks")
        raise HTTPException(status_code=500, detail=str(e))

def process_validation_feedback(feedback: ValidationFeedback) -> bool:
    """Process validation feedback from human reviewers and store corrections."""
    try:
        # Generate corrections file path
        corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{feedback.pdf_id}_corrections.json"
        
        # Load existing corrections or initialize empty list
        if corrections_file.exists():
            with open(corrections_file, "r") as f:
                corrections = json.load(f)
        else:
            corrections = []
        
        # Add new correction
        correction = {
            "task_id": feedback.task_id,
            "table_id": feedback.table_id,
            "validation_status": feedback.validation_status,
            "table_data": feedback.table_data,
            "table_bbox": feedback.table_bbox,
            "merge_target": feedback.merge_target,
            "comment": feedback.comment,
            "extraction_quality": feedback.extraction_quality,
            "timestamp": str(uuid.uuid4())
        }
        
        corrections.append(correction)
        
        # Save updated corrections
        with open(corrections_file, "w") as f:
            json.dump(corrections, f, indent=2)
        
        logger.info(f"Saved correction for table {feedback.table_id} in PDF {feedback.pdf_id}")
        return True
    except Exception as e:
        logger.exception("Error processing validation feedback")
        raise HTTPException(status_code=500, detail=str(e))

def apply_corrections_to_extraction(pdf_id: str, background_tasks: BackgroundTasks) -> bool:
    """Apply human-validated corrections to the extraction results."""
    try:
        # Path to corrections and original extraction output
        corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
        output_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_structured.json"
        
        if not corrections_file.exists() or not output_file.exists():
            logger.error(f"Missing corrections or output file for PDF {pdf_id}")
            raise HTTPException(status_code=404, detail="Corrections or output file not found")
        
        # Load corrections and original extraction
        with open(corrections_file, "r") as f:
            corrections = json.load(f)
        
        with open(output_file, "r") as f:
            extraction = json.load(f)
        
        # Schedule the background task
        logger.info(f"Scheduling correction application for PDF {pdf_id}")
        background_tasks.add_task(
            _apply_corrections_in_background, 
            pdf_id=pdf_id,
            corrections=corrections,
            extraction=extraction
        )
        
        return True
    except Exception as e:
        logger.exception("Error applying corrections")
        raise HTTPException(status_code=500, detail=str(e))

async def _apply_corrections_in_background(pdf_id: str, corrections: List[Dict], extraction: List[Dict]) -> bool:
    """Background task to apply corrections to extraction results."""
    try:
        tables_to_update = {}
        tables_to_remove = []
        tables_to_add = []
        merges = {}
        
        # 1. Process each correction
        for correction in corrections:
            table_id = correction["table_id"]
            status = correction["validation_status"]
            
            if status == "Approve":
                # No changes needed
                continue
            
            elif status == "Edit":
                # Store edited table data
                if correction.get("table_data"):
                    tables_to_update[table_id] = correction["table_data"]
            
            elif status == "Reject":
                # Mark table for removal
                tables_to_remove.append(table_id)
            
            elif status == "Add Table":
                # Store new table data
                if correction.get("table_data") and correction.get("table_bbox"):
                    tables_to_add.append({
                        "id": f"manual_{len(tables_to_add)}",
                        "data": correction["table_data"],
                        "bbox": correction["table_bbox"],
                        "page": int(correction.get("page", 1))
                    })
            
            elif status == "Merge":
                # Store merge information
                if correction.get("merge_target"):
                    target = correction["merge_target"]
                    if target not in merges:
                        merges[target] = []
                    merges[target].append(table_id)
        
        # 2. Apply modifications to extraction results
        modified_extraction = []
        tables_already_processed = set()
        
        # Process tables
        for item in extraction:
            # Skip non-table items
            if item.get("type") != "table":
                modified_extraction.append(item)
                continue
                
            table_id = item.get("id")
            if not table_id:
                modified_extraction.append(item)
                continue
                
            # Skip tables marked for removal
            if table_id in tables_to_remove:
                continue
                
            # Skip tables that are part of a merge (will handle them later)
            if any(table_id in source_ids for source_ids in merges.values()):
                continue
                
            # Apply edits if needed
            if table_id in tables_to_update:
                item["headers"] = tables_to_update[table_id]["headers"]
                item["rows"] = tables_to_update[table_id]["rows"]
                item["metadata"]["human_validated"] = True
                item["metadata"]["validation_date"] = str(uuid.uuid4())  # Use as timestamp
                
            # Add to modified extraction and mark as processed
            modified_extraction.append(item)
            tables_already_processed.add(table_id)
            
        # Handle merges
        for target, sources in merges.items():
            # Find target table
            target_table = None
            for item in extraction:
                if item.get("type") == "table" and item.get("id") == target:
                    target_table = item.copy()
                    break
                    
            if target_table and target not in tables_already_processed:
                # Process sources if target exists
                for source in sources:
                    for item in extraction:
                        if item.get("type") == "table" and item.get("id") == source:
                            # Add rows from source to target
                            target_table["rows"].extend(item.get("rows", []))
                
                # Mark as merged
                target_table["metadata"]["merged"] = True
                target_table["metadata"]["human_validated"] = True
                target_table["metadata"]["merged_sources"] = sources
                target_table["metadata"]["validation_date"] = str(uuid.uuid4())
                
                # Add to modified extraction
                modified_extraction.append(target_table)
                tables_already_processed.add(target)
        
        # Handle additions
        for new_table in tables_to_add:
            table_item = {
                "type": "table",
                "id": new_table["id"],
                "headers": new_table["data"]["headers"],
                "rows": new_table["data"]["rows"],
                "page": new_table["page"],
                "metadata": {
                    "extraction_method": "manual",
                    "human_validated": True,
                    "confidence": 1.0,
                    "validation_date": str(uuid.uuid4())
                }
            }
            
            if new_table.get("bbox"):
                table_item["bbox"] = new_table["bbox"]
                
            modified_extraction.append(table_item)
        
        # Save corrected extraction
        corrected_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_corrected.json"
        with open(corrected_file, "w") as f:
            json.dump(modified_extraction, f, indent=2)
        
        logger.info(f"Applied corrections to extraction results for PDF {pdf_id}")
        return True
    except Exception as e:
        logger.exception(f"Error in background correction task for PDF {pdf_id}")
        return False

async def conversion_progress_generator(pdf_path: str, params: Dict[str, Any]):
    """Generator for streaming conversion progress via SSE."""
    try:
        # Initial event
        yield {
            "event": "start",
            "data": {"message": f"Starting conversion for {Path(pdf_path).name}"}
        }
        
        # Start conversion as a background task
        task = asyncio.create_task(
            asyncio.to_thread(
                convert_pdf_to_json,
                pdf_path=pdf_path,
                **params
            )
        )
        
        # Simulate progress updates (in reality you'd get progress from the converter)
        total_elements = 20  # Placeholder value
        for i in range(1, total_elements + 1):
            yield {
                "event": "progress",
                "data": {"elements_extracted": i, "total": total_elements}
            }
            await asyncio.sleep(0.2)  # Simulate processing time
        
        # Wait for conversion to complete
        result = await task
        
        # Final event with results
        yield {
            "event": "complete",
            "data": {"message": "Conversion complete", "data": result}
        }
    except Exception as e:
        logger.exception(f"Error in conversion progress generator: {str(e)}")
        yield {
            "event": "error",
            "data": {"message": f"Error during conversion: {str(e)}"}
        }

# API Endpoints
@app.post("/convert", response_model=ConversionResponse)
async def convert_pdf_endpoint(
    file: UploadFile = File(...),
    repo_link: str = Form(...),
    use_marker_markdown: bool = Form(False),
    force_qwen: bool = Form(False),
    output_dir: str = Form(DEFAULT_OUTPUT_DIR),
    corrections_dir: str = Form(DEFAULT_CORRECTIONS_DIR)
):
    """
    Converts an uploaded PDF to structured JSON.

    Args:
        file: PDF file to process.
        repo_link: Repository link for metadata.
        use_marker_markdown: Use Marker's Markdown output if True.
        force_qwen: Force Qwen-VL processing if True.
        output_dir: Directory for JSON output.
        corrections_dir: Directory for correction files.

    Returns:
        ConversionResponse with status, message, and extracted data.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Check directory permissions
    for dir_path in [output_dir, corrections_dir, DEFAULT_UPLOADS_DIR]:
        dir_obj = Path(dir_path)
        try:
            dir_obj.mkdir(parents=True, exist_ok=True)
            if not os.access(dir_obj, os.W_OK):
                raise PermissionError(f"Write permission denied for {dir_path}")
        except Exception as e:
            logger.error(f"Failed to access directory {dir_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Directory access failed: {str(e)}")

    # Save to uploads directory for Label Studio access
    upload_path = Path(DEFAULT_UPLOADS_DIR) / file.filename
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    logger.info(f"Processing file: {file.filename}, repo_link: {repo_link}")
    try:
        result = convert_pdf_to_json(
            pdf_path=str(upload_path),
            repo_link=repo_link,
            output_dir=output_dir,
            use_marker_markdown=use_marker_markdown,
            corrections_dir=corrections_dir,
            force_qwen=force_qwen
        )
        
        if not result:
            logger.warning("No data extracted from PDF.")
            return ConversionResponse(
                status="success",
                message="No content extracted from the PDF.",
                data=[]
            )
            
        # Save the extraction result
        pdf_id = Path(upload_path).stem
        output_file = Path(output_dir) / f"{pdf_id}_structured.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Extracted {len(result)} elements from {file.filename}")
        return ConversionResponse(
            status="success",
            message=f"PDF converted successfully. Extracted {len(result)} elements.",
            data=result
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.post("/stream/convert")
async def stream_convert_endpoint(
    file: UploadFile = File(...),
    repo_link: str = Form(...),
    use_marker_markdown: bool = Form(False),
    force_qwen: bool = Form(False),
    output_dir: str = Form(DEFAULT_OUTPUT_DIR),
    corrections_dir: str = Form(DEFAULT_CORRECTIONS_DIR)
):
    """
    Streams the conversion progress of a PDF file to structured JSON via SSE.

    Args:
        file: PDF file to process.
        repo_link: Repository link for metadata.
        use_marker_markdown: Use Marker's Markdown output if True.
        force_qwen: Force Qwen-VL processing if True.
        output_dir: Directory for JSON output.
        corrections_dir: Directory for correction files.

    Returns:
        EventSourceResponse with streaming progress and results.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Check directory permissions
    for dir_path in [output_dir, corrections_dir, DEFAULT_UPLOADS_DIR]:
        dir_obj = Path(dir_path)
        try:
            dir_obj.mkdir(parents=True, exist_ok=True)
            if not os.access(dir_obj, os.W_OK):
                raise PermissionError(f"Write permission denied for {dir_path}")
        except Exception as e:
            logger.error(f"Failed to access directory {dir_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Directory access failed: {str(e)}")

    # Save to uploads directory
    upload_path = Path(DEFAULT_UPLOADS_DIR) / file.filename
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # Prepare conversion parameters
    params = {
        "repo_link": repo_link,
        "output_dir": output_dir,
        "use_marker_markdown": use_marker_markdown,
        "corrections_dir": corrections_dir,
        "force_qwen": force_qwen
    }

    logger.info(f"Streaming conversion for {file.filename}, repo_link: {repo_link}")
    
    # Return SSE response
    return EventSourceResponse(
        conversion_progress_generator(str(upload_path), params)
    )

@app.get("/status", response_model=StatusResponse)
async def status_endpoint():
    """
    Checks the API's status.

    Returns:
        StatusResponse with server status.
    """
    logger.info("Status check requested.")
    return StatusResponse(
        status="success",
        message="PDF to JSON Converter API is running with Label Studio integration."
    )

# Label Studio integration endpoints
@app.post("/label-studio/create-project", response_model=ProjectResponse)
async def create_project_endpoint(project_name: str = Form(...)):
    """
    Creates a new Label Studio project for PDF table validation.

    Args:
        project_name: Name for the new project.

    Returns:
        ProjectResponse with project ID and status message.
    """
    ensure_directories_exist()
    project_id = create_label_studio_project(project_name)
    return ProjectResponse(
        project_id=project_id, 
        message="Project created successfully"
    )

@app.post("/label-studio/create-tasks/{pdf_id}", response_model=TasksResponse)
async def create_tasks_endpoint(pdf_id: str, project_id: int = Form(...)):
    """
    Creates tasks in Label Studio for human validation of extracted tables.

    Args:
        pdf_id: ID of the PDF file (filename without extension).
        project_id: Label Studio project ID.

    Returns:
        TasksResponse with task count and status message.
    """
    # Path to extraction output
    output_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_structured.json"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Extraction output not found")
    
    # Load extraction output
    with open(output_file, "r") as f:
        extraction = json.load(f)
    
    # Filter for table elements
    tables = []
    for item in extraction:
        if item.get("type") == "table":
            tables.append(TableInfo(
                id=item.get("id", f"table_{len(tables)}"),
                page=item.get("page", 1),
                caption=item.get("caption"),
                headers=item.get("headers", []),
                rows=item.get("rows", []),
                metadata=item.get("metadata", {}),
                bbox=item.get("bbox")
            ))
    
    if not tables:
        logger.warning(f"No tables found in extraction for PDF {pdf_id}")
        return TasksResponse(
            pdf_id=pdf_id,
            project_id=project_id,
            task_count=0,
            message="No tables found for validation"
        )
    
    # Create tasks in Label Studio
    task_count = create_label_studio_tasks(pdf_id, project_id, tables)
    
    return TasksResponse(
        pdf_id=pdf_id,
        project_id=project_id,
        task_count=task_count,
        message=f"Created {task_count} validation tasks"
    )

@app.post("/label-studio/feedback", response_model=FeedbackResponse)
async def submit_feedback_endpoint(feedback: ValidationFeedback):
    """
    Submits validation feedback from human reviewers.

    Args:
        feedback: Validation feedback data.

    Returns:
        FeedbackResponse with status message.
    """
    result = process_validation_feedback(feedback)
    return FeedbackResponse(
        pdf_id=feedback.pdf_id,
        table_id=feedback.table_id,
        status="success" if result else "error",
        message="Feedback processed successfully" if result else "Error processing feedback"
    )

@app.post("/label-studio/apply-corrections/{pdf_id}", response_model=CorrectionResponse)
async def apply_corrections_endpoint(pdf_id: str, background_tasks: BackgroundTasks):
    """
    Applies human-validated corrections to the extraction results.

    Args:
        pdf_id: ID of the PDF file (filename without extension).
        background_tasks: FastAPI background tasks.

    Returns:
        CorrectionResponse with status message.
    """
    result = apply_corrections_to_extraction(pdf_id, background_tasks)
    return CorrectionResponse(
        pdf_id=pdf_id,
        status="scheduled" if result else "error",
        message="Correction application scheduled" if result else "Error scheduling correction application"
    )

@app.get("/label-studio/correction-status/{pdf_id}", response_model=CorrectionStatusResponse)
async def correction_status_endpoint(pdf_id: str):
    """
    Checks the status of correction application.

    Args:
        pdf_id: ID of the PDF file (filename without extension).

    Returns:
        CorrectionStatusResponse with status message.
    """
    corrected_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_corrected.json"
    
    if corrected_file.exists():
        return CorrectionStatusResponse(
            pdf_id=pdf_id,
            status="completed",
            message="Corrections have been applied"
        )
    
    corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
    if corrections_file.exists():
        return CorrectionStatusResponse(
            pdf_id=pdf_id,
            status="pending",
            message="Corrections received but not yet applied"
        )
    
    return CorrectionStatusResponse(
        pdf_id=pdf_id,
        status="not_found",
        message="No corrections found for this PDF"
    )

def usage_function():
    """
    Simulates API usage by running a conversion.

    Returns:
        dict: Simulated API response.
    """
    sample_pdf = "input/BHT_CV32A65X.pdf"
    repo_link = "https://github.com/example/repo"
    try:
        result = convert_pdf_to_json(sample_pdf, repo_link)
        return {
            "status": "success",
            "message": "PDF converted successfully.",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "data": []
        }

if __name__ == "__main__":
    fix_sys_path(__file__)
    logger.info("Testing API usage function...")
    result = usage_function()
    print("API Usage Function Result:")
    print(json.dumps(result, indent=2))
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)