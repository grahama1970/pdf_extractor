"""
Label Studio integration for human-in-the-loop PDF table extraction validation.

This module provides APIs for:
1. Creating Label Studio tasks from PDF extraction results
2. Processing validation feedback from human reviewers
3. Applying corrections to the extraction pipeline

Integration follows standards from VALIDATION_REQUIREMENTS.md and JSON_SCHEMA_FORMAT.md
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for file paths
CORRECTIONS_DIR = os.getenv("CORRECTIONS_DIR", "corrections")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://labelstudio:8080/api")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN", "")

# Initialize router
router = APIRouter(tags=["Label Studio Integration"])

# Models for task creation and validation
class TableInfo(BaseModel):
    id: str
    page: int
    caption: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]
    metadata: Dict[str, Any]
    bbox: Optional[List[float]] = None

class ValidationFeedback(BaseModel):
    task_id: str
    pdf_id: str
    table_id: str
    validation_status: str
    table_data: Optional[Dict] = None
    table_bbox: Optional[List[float]] = None
    merge_target: Optional[str] = None
    comment: Optional[str] = None


# Helper functions for Label Studio integration
def ensure_directories_exist():
    """Ensure all required directories exist."""
    for directory in [CORRECTIONS_DIR, OUTPUT_DIR, UPLOADS_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def create_label_studio_project(project_name: str) -> int:
    """Create a new Label Studio project for PDF table validation."""
    try:
        # Read labeling configuration from file
        with open("labeling_config.xml", "r") as f:
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
                "parsed_label_config": labeling_config
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


def create_label_studio_tasks(pdf_id: str, project_id: int, extracted_tables: List[TableInfo]):
    """Create tasks in Label Studio for human validation of extracted tables."""
    try:
        tasks = []
        
        for table in extracted_tables:
            # Prepare table data in the expected format for Label Studio
            task_data = {
                "pdf_id": pdf_id,
                "pdf_page_url": f"/data/local-files/?d={UPLOADS_DIR}/{pdf_id}.pdf#page={table.page}",
                "table_id": table.id,
                "source": table.metadata.get("extraction_method", "unknown"),
                "needs_review": True,
                "table_data": {
                    "headers": table.headers,
                    "rows": table.rows
                },
                "bbox": table.bbox
            }
            
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
                corrections_file = Path(CORRECTIONS_DIR) / f"{pdf_id}_tasks.json"
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


def process_validation_feedback(feedback: ValidationFeedback):
    """Process validation feedback from human reviewers and store corrections."""
    try:
        # Generate corrections file path
        corrections_file = Path(CORRECTIONS_DIR) / f"{feedback.pdf_id}_corrections.json"
        
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


def apply_corrections_to_extraction(pdf_id: str, background_tasks: BackgroundTasks):
    """Apply human-validated corrections to the extraction results."""
    try:
        # Path to corrections and original extraction output
        corrections_file = Path(CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
        output_file = Path(OUTPUT_DIR) / f"{pdf_id}_structured.json"
        
        if not corrections_file.exists() or not output_file.exists():
            logger.error(f"Missing corrections or output file for PDF {pdf_id}")
            raise HTTPException(status_code=404, detail="Corrections or output file not found")
        
        # Load corrections and original extraction
        with open(corrections_file, "r") as f:
            corrections = json.load(f)
        
        with open(output_file, "r") as f:
            extraction = json.load(f)
        
        # This would be the task to apply corrections
        # For now, just log that we would do this
        logger.info(f"Scheduling correction application for PDF {pdf_id}")
        
        # Here you would implement the logic to:
        # 1. Process each correction based on validation_status
        # 2. Update tables in the extraction results
        # 3. Handle merges, edits, rejections, and additions
        # 4. Save the corrected extraction
        
        # Add to background tasks
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


async def _apply_corrections_in_background(pdf_id: str, corrections: List[Dict], extraction: Dict):
    """Background task to apply corrections to extraction results."""
    try:
        # Process each correction
        tables_to_update = {}
        tables_to_remove = []
        tables_to_add = []
        merges = {}
        
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
                        "bbox": correction["table_bbox"]
                    })
            
            elif status == "Merge":
                # Store merge information
                if correction.get("merge_target"):
                    target = correction["merge_target"]
                    if target not in merges:
                        merges[target] = []
                    merges[target].append(table_id)
        
        # Apply updates to extraction results
        # 1. First, handle updates
        for i, table in enumerate(extraction.get("tables", [])):
            if table["id"] in tables_to_update:
                extraction["tables"][i]["headers"] = tables_to_update[table["id"]]["headers"]
                extraction["tables"][i]["rows"] = tables_to_update[table["id"]]["rows"]
                extraction["tables"][i]["metadata"]["human_validated"] = True
        
        # 2. Then, handle removals
        extraction["tables"] = [
            table for table in extraction.get("tables", [])
            if table["id"] not in tables_to_remove
        ]
        
        # 3. Handle additions
        for new_table in tables_to_add:
            extraction["tables"].append({
                "id": new_table["id"],
                "headers": new_table["data"]["headers"],
                "rows": new_table["data"]["rows"],
                "page": 1,  # Would need to extract from bbox or task data
                "metadata": {
                    "extraction_method": "manual",
                    "human_validated": True,
                    "confidence": 1.0
                }
            })
        
        # 4. Handle merges (simplified - real implementation would be more complex)
        for target, sources in merges.items():
            # Find target table
            target_table = None
            for table in extraction.get("tables", []):
                if table["id"] == target:
                    target_table = table
                    break
            
            if target_table:
                # Merge source tables into target
                # This is a simplified example - actual merging logic would be more complex
                for source in sources:
                    for table in extraction.get("tables", []):
                        if table["id"] == source:
                            # Add rows from source to target
                            target_table["rows"].extend(table["rows"])
                            # Mark source for removal
                            tables_to_remove.append(source)
            
            # Remove merged source tables
            extraction["tables"] = [
                table for table in extraction.get("tables", [])
                if table["id"] not in tables_to_remove
            ]
        
        # Save corrected extraction
        corrected_file = Path(OUTPUT_DIR) / f"{pdf_id}_corrected.json"
        with open(corrected_file, "w") as f:
            json.dump(extraction, f, indent=2)
        
        logger.info(f"Applied corrections to extraction results for PDF {pdf_id}")
        return True
    except Exception as e:
        logger.exception(f"Error in background correction task for PDF {pdf_id}")
        return False


# API Endpoints
@router.post("/label-studio/create-project")
async def create_project(project_name: str = Form(...)):
    """Create a new Label Studio project for PDF table validation."""
    ensure_directories_exist()
    project_id = create_label_studio_project(project_name)
    return {"project_id": project_id, "message": "Project created successfully"}


@router.post("/label-studio/create-tasks/{pdf_id}")
async def create_tasks(pdf_id: str, project_id: int = Form(...)):
    """Create tasks in Label Studio for human validation of extracted tables."""
    # Path to extraction output
    output_file = Path(OUTPUT_DIR) / f"{pdf_id}_structured.json"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Extraction output not found")
    
    # Load extraction output
    with open(output_file, "r") as f:
        extraction = json.load(f)
    
    # Convert to TableInfo objects
    tables = []
    for table in extraction.get("tables", []):
        tables.append(TableInfo(
            id=table["id"],
            page=table["page"],
            caption=table.get("caption"),
            headers=table["headers"],
            rows=table["rows"],
            metadata=table["metadata"],
            bbox=table.get("bbox")
        ))
    
    # Create tasks in Label Studio
    task_count = create_label_studio_tasks(pdf_id, project_id, tables)
    
    return {
        "pdf_id": pdf_id,
        "project_id": project_id,
        "task_count": task_count,
        "message": f"Created {task_count} validation tasks"
    }


@router.post("/label-studio/feedback")
async def submit_feedback(feedback: ValidationFeedback):
    """Submit validation feedback from human reviewers."""
    result = process_validation_feedback(feedback)
    return {
        "pdf_id": feedback.pdf_id,
        "table_id": feedback.table_id,
        "status": "success" if result else "error",
        "message": "Feedback processed successfully" if result else "Error processing feedback"
    }


@router.post("/label-studio/apply-corrections/{pdf_id}")
async def apply_corrections(pdf_id: str, background_tasks: BackgroundTasks):
    """Apply human-validated corrections to the extraction results."""
    result = apply_corrections_to_extraction(pdf_id, background_tasks)
    return {
        "pdf_id": pdf_id,
        "status": "scheduled" if result else "error",
        "message": "Correction application scheduled" if result else "Error scheduling correction application"
    }


@router.get("/label-studio/correction-status/{pdf_id}")
async def correction_status(pdf_id: str):
    """Check the status of correction application."""
    corrected_file = Path(OUTPUT_DIR) / f"{pdf_id}_corrected.json"
    
    if corrected_file.exists():
        return {
            "pdf_id": pdf_id,
            "status": "completed",
            "message": "Corrections have been applied"
        }
    
    corrections_file = Path(CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
    if corrections_file.exists():
        return {
            "pdf_id": pdf_id,
            "status": "pending",
            "message": "Corrections received but not yet applied"
        }
    
    return {
        "pdf_id": pdf_id,
        "status": "not_found",
        "message": "No corrections found for this PDF"
    }


# Include additional endpoints as needed