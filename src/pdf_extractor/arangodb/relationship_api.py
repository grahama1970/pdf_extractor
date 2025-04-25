# src/pdf_extractor/arangodb/relationship_api.py
import sys
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from loguru import logger
from arango.database import StandardDatabase
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME, EDGE_COLLECTION_NAME, CONFIDENCE_SCORE_RANGE, RATIONALE_MIN_LENGTH
)
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database, ensure_edge_collection, ensure_graph

def validate_rationale(rationale: str) -> bool:
    """Check if rationale is valid (non-empty, >=50 chars)."""
    if not rationale or len(rationale) < RATIONALE_MIN_LENGTH:
        logger.error(f"Rationale too short: {len(rationale)} < {RATIONALE_MIN_LENGTH}")
        return False
    return True

def validate_confidence_score(score: int) -> bool:
    """Check if confidence score is valid (integer, 1-5)."""
    if not isinstance(score, int) or score < CONFIDENCE_SCORE_RANGE[0] or score > CONFIDENCE_SCORE_RANGE[1]:
        logger.error(f"Invalid confidence score: {score}, must be {CONFIDENCE_SCORE_RANGE}")
        return False
    return True

def add_relationship(
    db: StandardDatabase,
    from_doc_key: str,
    to_doc_key: str,
    rationale: str,
    relationship_type: str,
    confidence_score: int,
    attributes: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Add a relationship between documents."""
    from_id = f"{COLLECTION_NAME}/{from_doc_key}"
    to_id = f"{COLLECTION_NAME}/{to_doc_key}"
    logger.info(f"Adding relationship: {from_id} -> {to_id}")
    
    if not validate_rationale(rationale) or not validate_confidence_score(confidence_score):
        return None
    
    edge_data = {
        "_from": from_id,
        "_to": to_id,
        "rationale": rationale,
        "type": relationship_type,
        "confidence_score": confidence_score,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if attributes:
        edge_data.update(attributes)
    
    try:
        edge_collection = db.collection(EDGE_COLLECTION_NAME)
        meta = edge_collection.insert(edge_data, sync=True)
        logger.info(f"Created relationship: _key={meta['_key']}")
        return edge_data
    except Exception as e:
        logger.error(f"Cannot add relationship: {e}")
        return None

def delete_relationship(db: StandardDatabase, edge_key: str) -> bool:
    """Delete a relationship."""
    logger.info(f"Deleting relationship: {edge_key}")
    try:
        edge_collection = db.collection(EDGE_COLLECTION_NAME)
        return edge_collection.delete(edge_key, ignore_missing=True)
    except Exception as e:
        logger.error(f"Cannot delete relationship: {e}")
        return False

def get_relationships(
    db: StandardDatabase,
    doc_key: str,
    direction: str = "ANY",
    types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Get relationships for a document."""
    doc_id = f"{COLLECTION_NAME}/{doc_key}"
    logger.info(f"Getting relationships for: {doc_id}")
    try:
        if direction not in ["OUTBOUND", "INBOUND", "ANY"]:
            raise ValueError(f"Invalid direction: {direction}")
        
        filter_clause = ""
        if types:
            type_list = ", ".join([f"'{t}'" for t in types])
            filter_clause = f"FILTER e.type IN [{type_list}]"
        
        aql = f"""
        FOR v, e IN 1..1 {direction} @start_vertex GRAPH @graph_name
        {filter_clause}
        RETURN e
        """
        bind_vars = {"start_vertex": doc_id, "graph_name": "document_graph"}
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        return list(cursor)
    except Exception as e:
        logger.error(f"Cannot get relationships: {e}")
        return []

def validate_relationship(extracted_data, fixture_path):
    """Validate relationship against fixture."""
    validation_failures = {}
    try:
        with open(fixture_path, "r") as f:
            expected = json.load(f)
        
        # Structural validation
        required_fields = ["_from", "_to", "type", "rationale", "confidence_score"]
        for field in required_fields:
            if field not in extracted_data:
                validation_failures[field] = {
                    "expected": "present",
                    "actual": "missing"
                }
        
        # Content validation
        if extracted_data.get("type") != expected.get("type"):
            validation_failures["type"] = {
                "expected": expected["type"],
                "actual": extracted_data.get("type")
            }
        if not validate_rationale(extracted_data.get("rationale", "")):
            validation_failures["rationale"] = {
                "expected": f">={RATIONALE_MIN_LENGTH} chars",
                "actual": len(extracted_data.get("rationale", ""))
            }
        if not validate_confidence_score(extracted_data.get("confidence_score", 0)):
            validation_failures["confidence_score"] = {
                "expected": f"{CONFIDENCE_SCORE_RANGE}",
                "actual": extracted_data.get("confidence_score")
            }
        
        return len(validation_failures) == 0, validation_failures
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False, {"validation_error": str(e)}

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    client = connect_arango()
    db = ensure_database(client)
    ensure_edge_collection(db)
    ensure_graph(db)
    
    from_key = f"test_{uuid.uuid4().hex[:8]}"
    to_key = f"test_{uuid.uuid4().hex[:8]}"
    rationale = "Documents share database optimization techniques, confirmed via Perplexity."
    rel_type = "SIMILAR"
    confidence = 3
    
    vertex_collection = db.collection(COLLECTION_NAME)
    vertex_collection.insert({"_key": from_key, "content": "Test doc 1", "tags": ["database"]})
    vertex_collection.insert({"_key": to_key, "content": "Test doc 2", "tags": ["database"]})
    
    edge_data = add_relationship(db, from_key, to_key, rationale, rel_type, confidence)
    if not edge_data:
        print("❌ Failed to add relationship")
        sys.exit(1)
    
    validation_passed, validation_failures = validate_relationship(edge_data, "src/test_fixtures/relationship_expected.json")
    
    if validation_passed:
        print("✅ Relationship API validation passed")
    else:
        print("❌ VALIDATION FAILED - Relationship does not match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
    
    # Cleanup
    edges = get_relationships(db, from_key)
    for edge in edges:
        delete_relationship(db, edge["_key"])
    vertex_collection.delete(from_key)
    vertex_collection.delete(to_key)
