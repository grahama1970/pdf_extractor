# src/pdf_extractor/arangodb/search_api/graph_traverse.py
import sys
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger
from arango.database import StandardDatabase
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME, GRAPH_NAME
)
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database

def graph_traverse(
    db: StandardDatabase,
    start_vertex_key: str,
    min_depth: int = 1,
    max_depth: int = 1,
    direction: str = "ANY",
    relationship_types: Optional[List[str]] = None,
    vertex_filter: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Traverse the graph from a start vertex and return the connecting vertices.
    
    Args:
        db: ArangoDB database
        start_vertex_key: Key of the starting vertex
        min_depth: Minimum traversal depth
        max_depth: Maximum traversal depth
        direction: Direction of traversal (OUTBOUND, INBOUND, ANY)
        relationship_types: Optional list of relationship types to filter
        vertex_filter: Optional filter for vertices 
        limit: Maximum number of results to return
        
    Returns:
        Dictionary with results and metadata
    """
    logger.info(f"Traversing graph from {start_vertex_key} (depth: {min_depth}..{max_depth}, direction: {direction})")
    
    # Validate parameters
    if direction not in ["OUTBOUND", "INBOUND", "ANY"]:
        logger.error(f"Invalid direction: {direction}")
        return {"results": [], "count": 0, "error": f"Invalid direction: {direction}"}
    
    if min_depth < 0 or max_depth < min_depth:
        logger.error(f"Invalid depth range: {min_depth}..{max_depth}")
        return {"results": [], "count": 0, "error": f"Invalid depth range: {min_depth}..{max_depth}"}
    
    try:
        # Construct AQL query
        start_vertex = f"{COLLECTION_NAME}/{start_vertex_key}"
        
        # Build edge filter if relationship types are provided
        edge_filter = ""
        if relationship_types:
            type_list = ", ".join([f"'{t}'" for t in relationship_types])
            edge_filter = f"FILTER e.type IN [{type_list}]"
        
        # Build vertex filter if provided
        vert_filter = ""
        if vertex_filter:
            conditions = []
            for field, value in vertex_filter.items():
                if isinstance(value, str):
                    conditions.append(f"v.{field} == '{value}'")
                else:
                    conditions.append(f"v.{field} == {value}")
            
            if conditions:
                vert_filter = f"FILTER {' AND '.join(conditions)}"
        
        aql = f"""
        FOR v, e, p IN {min_depth}..{max_depth} {direction} @start_vertex GRAPH @graph_name
        {edge_filter}
        {vert_filter}
        LIMIT @limit
        RETURN {{
            "vertex": v,
            "edge": e,
            "path": p
        }}
        """
        
        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": GRAPH_NAME,
            "limit": limit
        }
        
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)
        
        logger.info(f"Traversal found {len(results)} results")
        return {
            "results": results,
            "count": len(results),
            "params": {
                "start_vertex": start_vertex_key,
                "min_depth": min_depth,
                "max_depth": max_depth,
                "direction": direction,
                "relationship_types": relationship_types
            }
        }
    except Exception as e:
        logger.error(f"Traversal error: {e}")
        return {"results": [], "count": 0, "error": str(e)}

def validate_traversal(traversal_results: Dict[str, Any], fixture_path: str) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate traversal results against expected fixture data.
    
    Args:
        traversal_results: Results from graph_traverse function
        fixture_path: Path to fixture JSON file with expected results
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    try:
        # Load fixture data
        with open(fixture_path, "r") as f:
            expected_data = json.load(f)
        
        # Structural validation
        if "results" not in traversal_results:
            validation_failures["missing_results"] = {
                "expected": "Results field present",
                "actual": "Results field missing"
            }
        
        if "count" not in traversal_results:
            validation_failures["missing_count"] = {
                "expected": "Count field present",
                "actual": "Count field missing"
            }
            
        if "params" not in traversal_results:
            validation_failures["missing_params"] = {
                "expected": "Params field present",
                "actual": "Params field missing"
            }
        
        # Content validation
        if "expected_count" in expected_data:
            if traversal_results.get("count", 0) < expected_data["expected_count"]:
                validation_failures["count_value"] = {
                    "expected": f">={expected_data['expected_count']}",
                    "actual": traversal_results.get("count", 0)
                }
        
        # Params validation
        if "expected_params" in expected_data and "params" in traversal_results:
            expected_params = expected_data["expected_params"]
            actual_params = traversal_results["params"]
            
            for param_name in expected_params:
                if param_name not in actual_params:
                    validation_failures[f"missing_param_{param_name}"] = {
                        "expected": f"{param_name} present",
                        "actual": f"{param_name} missing"
                    }
                elif actual_params[param_name] != expected_params[param_name]:
                    validation_failures[f"param_{param_name}"] = {
                        "expected": expected_params[param_name],
                        "actual": actual_params[param_name]
                    }
        
        # Result structure validation
        if "results" in traversal_results and len(traversal_results["results"]) > 0:
            first_result = traversal_results["results"][0]
            required_fields = ["vertex", "edge", "path"]
            
            for field in required_fields:
                if field not in first_result:
                    validation_failures[f"result_missing_{field}"] = {
                        "expected": f"{field} present in result",
                        "actual": f"{field} missing in result"
                    }
        
        return len(validation_failures) == 0, validation_failures
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, {"validation_error": {"expected": "Successful validation", "actual": str(e)}}

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Path to test fixture
    fixture_path = "src/test_fixtures/traversal_expected.json"
    
    try:
        # Initialize ArangoDB connection
        client = connect_arango()
        db = ensure_database(client)
        
        # Create test data or use existing
        vertex_collection = db.collection(COLLECTION_NAME)
        
        # Check if we have at least one vertex, create if needed
        if vertex_collection.count() == 0:
            test_key = "test_root_vertex"
            vertex_collection.insert({
                "_key": test_key,
                "content": "Root test vertex for traversal",
                "tags": ["test", "traversal"]
            })
        else:
            # Use the first document as starting point
            first_doc = next(vertex_collection.all())
            test_key = first_doc["_key"]
        
        # Run traversal
        results = graph_traverse(db, test_key, min_depth=1, max_depth=2)
        
        # Validate results
        validation_passed, validation_failures = validate_traversal(results, fixture_path)
        
        # Report validation status
        if validation_passed:
            print("✅ VALIDATION COMPLETE - All graph traversal results match expected values")
            sys.exit(0)
        else:
            print("❌ VALIDATION FAILED - Graph traversal results don't match expected values") 
            print(f"FAILURE DETAILS:")
            for field, details in validation_failures.items():
                print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
            print(f"Total errors: {len(validation_failures)} fields mismatched")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
