"""
Typer CLI for local PDF-to-JSON conversion with Label Studio integration.

This module provides a command-line interface to convert PDF files to structured
JSON, mirroring the FastAPI `/convert` endpoint. It integrates with the
`pdf_converter` module for core conversion logic and is designed for local
deployment, complementing the FastAPI server used in Docker containers.

Additionally, it provides commands for interacting with Label Studio for
human-in-the-loop validation of extracted content.

Dependencies:
- typer: For building the CLI.
- loguru: For logging.
- requests: For Label Studio API communication.

Usage:
    Convert a PDF:
    ```bash
    python cli.py convert sample.pdf \
        --repo-link https://github.com/example/repo \
        --output-dir output \
        --corrections-dir corrections
    ```

    Create Label Studio tasks for validation:
    ```bash
    python cli.py create-tasks sample.pdf \
        --project-id 1
    ```

    Apply Label Studio corrections:
    ```bash
    python cli.py apply-corrections sample
    ```

    With optional flags:
    ```bash
    python cli.py convert sample.pdf \
        --repo-link https://github.com/example/repo \
        --use-marker-markdown \
        --force-qwen
    ```
"""

import json
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import typer
from loguru import logger
import requests
import shutil
import sys

from pdf_extractor.config import (
    DEFAULT_OUTPUT_DIR, 
    DEFAULT_CORRECTIONS_DIR, 
    DEFAULT_UPLOADS_DIR,
    LABEL_STUDIO_URL,
    LABEL_STUDIO_TOKEN
)
from pdf_extractor.pdf_to_json_converter import convert_pdf_to_json
from pdf_extractor.utils import fix_sys_path

app = typer.Typer(
    name="pdf-extractor",
    help="CLI for converting PDFs to structured JSON with Label Studio integration.",
)

# Add Label Studio command group
label_studio_app = typer.Typer(
    name="label-studio",
    help="Commands for interacting with Label Studio for human-in-the-loop validation.",
)

app.add_typer(label_studio_app, name="label-studio")

# Helper functions
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
            typer.echo(f"Error: Failed to create Label Studio project: {response.text}")
            raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Error creating Label Studio project")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)

def create_label_studio_tasks(pdf_id: str, project_id: int) -> int:
    """Create tasks in Label Studio for human validation of extracted tables."""
    try:
        # Path to extraction output
        output_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_structured.json"
        
        if not output_file.exists():
            logger.error(f"Extraction output not found: {output_file}")
            typer.echo(f"Error: Extraction output not found for PDF {pdf_id}")
            raise typer.Exit(code=1)
        
        # Load extraction output
        with open(output_file, "r") as f:
            extraction = json.load(f)
        
        # Filter for table elements
        tables = []
        for item in extraction:
            if item.get("type") == "table":
                tables.append({
                    "id": item.get("id", f"table_{len(tables)}"),
                    "page": item.get("page", 1),
                    "caption": item.get("caption"),
                    "headers": item.get("headers", []),
                    "rows": item.get("rows", []),
                    "metadata": item.get("metadata", {}),
                    "bbox": item.get("bbox")
                })
        
        if not tables:
            logger.warning(f"No tables found in extraction for PDF {pdf_id}")
            typer.echo(f"Warning: No tables found for validation in PDF {pdf_id}")
            return 0
        
        # Create Label Studio tasks
        tasks = []
        
        for table in tables:
            # Prepare table data in the expected format for Label Studio
            task_data = {
                "pdf_id": pdf_id,
                "page": table["page"],
                "pdf_page_url": f"/data/upload/uploads/{pdf_id}.pdf#page={table['page']}",
                "table_id": table["id"],
                "source": table["metadata"].get("extraction_method", "unknown"),
                "table_data": {
                    "headers": table["headers"],
                    "rows": table["rows"]
                }
            }
            
            if table.get("bbox"):
                task_data["bbox"] = table["bbox"]
            
            # Create task in Label Studio
            tasks.append({
                "data": task_data,
                "meta": {
                    "pdf_id": pdf_id,
                    "table_id": table["id"],
                    "page": table["page"]
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
                typer.echo(f"Error: Failed to create Label Studio tasks: {response.text}")
                raise typer.Exit(code=1)
        
        return 0
    except Exception as e:
        logger.exception("Error creating Label Studio tasks")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)

def apply_corrections_to_extraction(pdf_id: str) -> bool:
    """Apply human-validated corrections to the extraction results."""
    try:
        # Path to corrections and original extraction output
        corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
        output_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_structured.json"
        
        if not corrections_file.exists():
            logger.error(f"Corrections file not found: {corrections_file}")
            typer.echo(f"Error: Corrections file not found for PDF {pdf_id}")
            raise typer.Exit(code=1)
            
        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            typer.echo(f"Error: Output file not found for PDF {pdf_id}")
            raise typer.Exit(code=1)
        
        # Load corrections and original extraction
        with open(corrections_file, "r") as f:
            corrections = json.load(f)
        
        with open(output_file, "r") as f:
            extraction = json.load(f)
        
        # Process corrections
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
        typer.echo(f"Successfully applied corrections to {pdf_id}")
        typer.echo(f"Corrected output saved to: {corrected_file}")
        return True
    except Exception as e:
        logger.exception(f"Error applying corrections to PDF {pdf_id}: {e}")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)

def check_correction_status(pdf_id: str) -> str:
    """Check the status of correction application."""
    corrected_file = Path(DEFAULT_OUTPUT_DIR) / f"{pdf_id}_corrected.json"
    
    if corrected_file.exists():
        return "completed"
    
    corrections_file = Path(DEFAULT_CORRECTIONS_DIR) / f"{pdf_id}_corrections.json"
    if corrections_file.exists():
        return "pending"
    
    return "not_found"


# CLI Commands
@app.command()
def convert(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to convert."),
    repo_link: str = typer.Option(
        "https://github.com/example/repo", help="Repository link for metadata."
    ),
    use_marker_markdown: bool = typer.Option(
        False, "--use-marker-markdown", help="Use Marker's Markdown output."
    ),
    force_qwen: bool = typer.Option(
        False, "--force-qwen", help="Force Qwen-VL processing."
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Directory for JSON output."
    ),
    corrections_dir: str = typer.Option(
        DEFAULT_CORRECTIONS_DIR, help="Directory for correction files."
    ),
    create_tasks: bool = typer.Option(
        False, "--create-tasks", help="Automatically create Label Studio tasks after conversion."
    ),
    project_id: Optional[int] = typer.Option(
        None, "--project-id", help="Label Studio project ID for task creation."
    ),
):
    """
    Converts a PDF to structured JSON, mirroring the FastAPI /convert endpoint.

    Args:
        pdf_path: Path to the PDF file.
        repo_link: Repository link for metadata.
        use_marker_markdown: Use Marker's Markdown output if True.
        force_qwen: Force Qwen-VL processing if True.
        output_dir: Directory for JSON output.
        corrections_dir: Directory for correction files.
        create_tasks: Create Label Studio tasks after conversion if True.
        project_id: Label Studio project ID for task creation.
    """
    # Ensure directories exist
    ensure_directories_exist()

    pdf_path_obj = Path(pdf_path).resolve()
    if not pdf_path_obj.is_file() or not pdf_path_obj.suffix.lower() == ".pdf":
        logger.error(f"Invalid PDF file: {pdf_path_obj}")
        typer.echo(f"Error: '{pdf_path_obj}' is not a valid PDF file.")
        raise typer.Exit(code=1)

    # Check directory permissions
    for dir_path in [output_dir, corrections_dir, DEFAULT_UPLOADS_DIR]:
        dir_obj = Path(dir_path)
        try:
            dir_obj.mkdir(parents=True, exist_ok=True)
            if not os.access(dir_obj, os.W_OK):
                raise PermissionError(f"Write permission denied for {dir_path}")
        except Exception as e:
            logger.error(f"Failed to access directory {dir_path}: {e}")
            typer.echo(f"Error: Directory access failed: {str(e)}")
            raise typer.Exit(code=1)

    # Copy PDF to uploads directory for Label Studio access
    upload_path = Path(DEFAULT_UPLOADS_DIR) / pdf_path_obj.name
    shutil.copy2(pdf_path_obj, upload_path)
    
    logger.info(f"Converting PDF: {pdf_path_obj}, repo_link: {repo_link}")
    try:
        result = convert_pdf_to_json(
            pdf_path=str(pdf_path_obj),
            repo_link=repo_link,
            output_dir=output_dir,
            use_marker_markdown=use_marker_markdown,
            corrections_dir=corrections_dir,
            force_qwen=force_qwen,
        )
        output_json = Path(output_dir) / f"{pdf_path_obj.stem}_structured.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Converted {pdf_path_obj.name}. Saved to {output_json}")
        typer.echo(f"Successfully converted '{pdf_path_obj.name}' to JSON.")
        typer.echo(f"Output saved to: {output_json}")
        typer.echo(f"Elements extracted: {len(result)}")
        
        # Create Label Studio tasks if requested
        if create_tasks:
            if not project_id:
                logger.error("Project ID is required to create Label Studio tasks")
                typer.echo("Error: Project ID is required to create Label Studio tasks")
                raise typer.Exit(code=1)
            
            task_count = create_label_studio_tasks(pdf_path_obj.stem, project_id)
            if task_count > 0:
                typer.echo(f"Created {task_count} Label Studio validation tasks")
            else:
                typer.echo("No tables found for validation")
                
    except Exception as e:
        logger.error(f"Conversion failed for '{pdf_path_obj}': {e}")
        typer.echo(f"Error: Conversion failed: {str(e)}")
        raise typer.Exit(code=1)

@label_studio_app.command("create-project")
def create_project(
    project_name: str = typer.Argument(..., help="Name for the new Label Studio project.")
):
    """
    Creates a new Label Studio project for PDF table validation.

    Args:
        project_name: Name for the new project.
    """
    ensure_directories_exist()
    project_id = create_label_studio_project(project_name)
    typer.echo(f"Successfully created Label Studio project: {project_id}")
    return project_id

@label_studio_app.command("create-tasks")
def create_tasks(
    pdf_id: str = typer.Argument(..., help="PDF ID (filename without extension)."),
    project_id: int = typer.Option(..., "--project-id", help="Label Studio project ID.")
):
    """
    Creates tasks in Label Studio for human validation of extracted tables.

    Args:
        pdf_id: PDF ID (filename without extension).
        project_id: Label Studio project ID.
    """
    ensure_directories_exist()
    task_count = create_label_studio_tasks(pdf_id, project_id)
    if task_count > 0:
        typer.echo(f"Successfully created {task_count} Label Studio validation tasks")
    else:
        typer.echo("No tables found for validation")
    return task_count

@label_studio_app.command("apply-corrections")
def apply_corrections(
    pdf_id: str = typer.Argument(..., help="PDF ID (filename without extension).")
):
    """
    Applies human-validated corrections to the extraction results.

    Args:
        pdf_id: PDF ID (filename without extension).
    """
    ensure_directories_exist()
    status = check_correction_status(pdf_id)
    
    if status == "not_found":
        typer.echo(f"No corrections found for PDF {pdf_id}")
        raise typer.Exit(code=1)
    elif status == "completed":
        typer.echo(f"Corrections already applied for PDF {pdf_id}")
        return True
        
    apply_corrections_to_extraction(pdf_id)
    return True

@label_studio_app.command("correction-status")
def correction_status(
    pdf_id: str = typer.Argument(..., help="PDF ID (filename without extension).")
):
    """
    Checks the status of correction application.

    Args:
        pdf_id: PDF ID (filename without extension).
    """
    status = check_correction_status(pdf_id)
    
    if status == "completed":
        typer.echo(f"Corrections have been applied for PDF {pdf_id}")
    elif status == "pending":
        typer.echo(f"Corrections received but not yet applied for PDF {pdf_id}")
    else:
        typer.echo(f"No corrections found for PDF {pdf_id}")
    
    return status

def usage_function() -> Dict[str, Any]:
    """
    Simulates CLI usage by running a conversion.

    Returns:
        dict: Simulated CLI result.
    """
    sample_pdf = "input/BHT_CV32A65X.pdf"
    repo_link = "https://github.com/example/repo"
    output_dir = DEFAULT_OUTPUT_DIR
    try:
        result = convert_pdf_to_json(
            pdf_path=sample_pdf,
            repo_link=repo_link,
            output_dir=output_dir,
            corrections_dir=DEFAULT_CORRECTIONS_DIR,
        )
        return {
            "status": "success",
            "message": f"Converted '{sample_pdf}' to JSON.",
            "output_file": f"{output_dir}/{Path(sample_pdf).stem}_structured.json",
            "elements_extracted": len(result),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "output_file": "",
            "elements_extracted": 0,
        }


if __name__ == "__main__":
    fix_sys_path(__file__)
    logger.info("Testing CLI usage function...")
    result = usage_function()
    print("CLI Usage Function Result:")
    print(json.dumps(result, indent=2))
    app()