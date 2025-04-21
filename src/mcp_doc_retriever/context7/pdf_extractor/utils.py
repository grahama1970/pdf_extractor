"""
Utility functions for PDF extractor.

This module provides common utilities used across the PDF extraction package,
including path management and geometry calculations.

Example Usage:
    # Fix Python path for imports
    >>> from mcp_doc_retriever.context7.pdf_extractor.utils import fix_sys_path
    >>> fix_sys_path()  # Adds project root to sys.path
    
    # Calculate IOU between bounding boxes
    >>> from mcp_doc_retriever.context7.pdf_extractor.utils import calculate_iou
    >>> box1 = [0.0, 0.0, 2.0, 2.0]  # 2x2 box at origin
    >>> box2 = [1.0, 1.0, 3.0, 3.0]  # 2x2 box overlapping by 1x1
    >>> iou = calculate_iou(box1, box2)
    >>> print(f"{iou:.2f}")  # Output: 0.14
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

def fix_sys_path(file_path: Optional[str] = None) -> None:
    """
    Adds the project root directory to sys.path to allow relative imports.
    This is particularly useful for CLI tools and standalone script execution.
    
    Args:
        file_path: Optional path to use as reference. If provided, uses this file's
                  location to determine project root. If None, uses current module location.
    """
    if file_path:
        current_dir = Path(file_path).resolve().parent
    else:
        current_dir = Path(__file__).resolve().parent
        
    project_root = current_dir
    while not (project_root / 'pyproject.toml').exists() and project_root != project_root.parent:
        project_root = project_root.parent
        
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate intersection over union (IOU) between two bounding boxes.
    Each box is [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right.
    
    Args:
        box1: First bounding box coordinates [x1, y1, x2, y2]
        box2: Second bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        float: IOU score between 0 and 1
    """
    # Get coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IOU = intersection / union
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area

if __name__ == "__main__":
    # Define expected results for validation
    EXPECTED_RESULTS = {
        "iou_partial_overlap": 1/7,  # Expected value for partial overlap test
        "iou_no_overlap": 0.0,      # Expected value for no overlap test
        "sys_path_updated": True,   # Expected project root to be in sys.path
        "pyproject_exists": True    # Expected pyproject.toml to exist in root
    }
    
    validation_passed = True
    actual_results = {}
    
    print("Running utils.py verification tests...\n")
    
    print("1. Testing IOU calculation:")
    print("--------------------------")
    # Test case 1: 25% overlap
    box1 = [0.0, 0.0, 2.0, 2.0]  # 2x2 box at origin
    box2 = [1.0, 1.0, 3.0, 3.0]  # 2x2 box overlapping by 1x1
    iou = calculate_iou(box1, box2)
    expected = EXPECTED_RESULTS["iou_partial_overlap"]
    
    print(f"Test 1 - Partial overlap:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box2}")
    print(f"  IOU: {iou:.3f} (expected: {expected:.3f})")
    
    try:
        assert abs(iou - expected) < 0.001, f"IOU calculation error: got {iou}, expected {expected}"
        actual_results["iou_partial_overlap"] = iou
        print(f"  ✓ Test passed: IOU value matches expected result")
    except AssertionError as e:
        actual_results["iou_partial_overlap"] = iou
        validation_passed = False
        print(f"  ✗ Test failed: {e}")
    
    # Test case 2: No overlap
    box3 = [3.0, 3.0, 4.0, 4.0]
    iou = calculate_iou(box1, box3)
    expected = EXPECTED_RESULTS["iou_no_overlap"]
    
    print(f"\nTest 2 - No overlap:")
    print(f"  Box 1: {box1}")
    print(f"  Box 2: {box3}")
    print(f"  IOU: {iou:.3f} (expected: {expected:.3f})")
    
    try:
        assert iou == expected, f"IOU should be {expected} for non-overlapping boxes, got {iou}"
        actual_results["iou_no_overlap"] = iou
        print(f"  ✓ Test passed: IOU value matches expected result")
    except AssertionError as e:
        actual_results["iou_no_overlap"] = iou
        validation_passed = False
        print(f"  ✗ Test failed: {e}")
    
    print("\n2. Testing path fixing:")
    print("----------------------")
    # Test basic path fixing
    original_path = sys.path.copy()
    fix_sys_path()
    print(f"Test 1 - Basic path fixing:")
    print(f"  Project root found: {sys.path[0]}")
    
    try:
        project_root_added = sys.path[0] != original_path[0] if original_path else True
        actual_results["sys_path_updated"] = project_root_added
        assert project_root_added, "Project root not added to sys.path"
        print(f"  ✓ Test passed: Project root added to sys.path")
    except AssertionError as e:
        actual_results["sys_path_updated"] = False
        validation_passed = False
        print(f"  ✗ Test failed: {e}")
    
    # Verify pyproject.toml exists
    pyproject_exists = (Path(sys.path[0]) / 'pyproject.toml').exists()
    actual_results["pyproject_exists"] = pyproject_exists
    print(f"\nTest 2 - Project structure verification:")
    print(f"  pyproject.toml exists: {pyproject_exists}")
    
    try:
        assert pyproject_exists == EXPECTED_RESULTS["pyproject_exists"], "pyproject.toml should exist in project root"
        print(f"  ✓ Test passed: Project structure verified")
    except AssertionError as e:
        validation_passed = False
        print(f"  ✗ Test failed: {e}")
    
    # Final validation check
    print("\nValidation Results:")
    for key, expected in EXPECTED_RESULTS.items():
        actual = actual_results.get(key)
        match = False
        
        if key.startswith("iou_"):
            # For floating point comparisons, use an epsilon
            match = abs(actual - expected) < 0.001 if actual is not None else False
        else:
            match = actual == expected
            
        print(f"  {key}: {'✓' if match else '✗'} Expected: {expected}, Got: {actual}")
        if not match:
            validation_passed = False
    
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - All results match expected values")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Results don't match expected values")
        print(f"Expected: {EXPECTED_RESULTS}")
        print(f"Got: {actual_results}")
        sys.exit(1)
