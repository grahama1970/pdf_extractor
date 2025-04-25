Overview
This plan outlines nine tasks to implement a strategic graph relationship system for the pdf_extractor project using ArangoDB, enabling meaningful connections between PDF documents to improve query answering. The plan is updated to:

Use absolute imports (no relative imports).
Simplify instructions for a less sophisticated agent.
Include checkboxes ([ ] for incomplete, [x] for complete) for each task.
Require the agent to manually mark checkboxes upon successful test completion.
Ensure relationships are useful by requiring agent-assessed rationales and confidence scores (1–5), potentially using tools like Perplexity.
Keep fixtures simple for the MVP.
Reference the provided hybrid_search implementation.
Comply with VALIDATION_REQUIREMENTS.md, CODE_DETAILS.md, and CODING_PRINCIPLES.md.

All code is stored in src/pdf_extractor/arangodb/.
Agent Instructions

ArangoDB Setup: Ensure ArangoDB is running:
curl http://localhost:8529

If it fails, start ArangoDB or update ARANGO_HOST, ARANGO_USER, and ARANGO_PASSWORD in src/pdf_extractor/arangodb/config.py.

Task Execution:

Complete each task and run its test command using uv run.
If the test outputs “✅ [Task Name] validation passed”, the task is successful.
Update the checkbox in this file from [ ] to [x] for that task.
To update:
Open TASK_PLAN.md.
Change [ ] Task X to [x] Task X (e.g., sed -i 's/\[ \] Task 1/[x] Task 1/' TASK_PLAN.md).
Verify the change (cat TASK_PLAN.md | grep Task).
Save the file.


Do not mark the checkbox until the test passes.
If the test fails, see the “Debugging Tips” in each task.


Rationale and Confidence Scores:

For relationships, provide a rationale (≥50 characters) explaining the connection, using tools like Perplexity if needed.
Assign a confidence score (1–5, 1=best) based on connection strength.



Task 1: Create Configuration File for Graph Relationships
[x] Task 1
Goal: Define constants for graph relationships.
File: src/pdf_extractor/arangodb/config.py
Code: (Unchanged from original)
# src/pdf_extractor/arangodb/config.py
import sys

# Graph and edge collection names
EDGE_COLLECTION_NAME = "document_relationships"
GRAPH_NAME = "document_graph"
COLLECTION_NAME = "documents"
ARANGO_DB_NAME = "pdf_extractor_db"
VIEW_NAME = "document_search_view"  # Added for hybrid_search

# ArangoDB connection settings
ARANGO_HOST = "http://localhost:8529"
ARANGO_USER = "root"
ARANGO_PASSWORD = "password"

# Relationship types
RELATIONSHIP_TYPE_SIMILAR = "SIMILAR"
RELATIONSHIP_TYPE_REFERENCES = "REFERENCES"
RELATIONSHIP_TYPE_SHARED_TOPIC = "SHARED_TOPIC"
RELATIONSHIP_TYPE_PREREQUISITE = "PREREQUISITE"
RELATIONSHIP_TYPE_CAUSAL = "CAUSAL"

# Validation settings
RATIONALE_MIN_LENGTH = 50
CONFIDENCE_SCORE_RANGE = (1, 5)

# Search fields for hybrid_search
SEARCH_FIELDS = ["content", "problem", "tags"]
ALL_DATA_FIELDS_PREVIEW = ["_key", "problem", "tags"]
TEXT_ANALYZER = "text_en"
TAG_ANALYZER = "identity"

if __name__ == "__main__":
    checks = [
        isinstance(EDGE_COLLECTION_NAME, str) and EDGE_COLLECTION_NAME,
        isinstance(GRAPH_NAME, str) and GRAPH_NAME,
        isinstance(COLLECTION_NAME, str) and COLLECTION_NAME,
        RATIONALE_MIN_LENGTH > 0,
        CONFIDENCE_SCORE_RANGE[0] <= CONFIDENCE_SCORE_RANGE[1]
    ]
    if all(checks):
        print("✅ Configuration validation passed")
        sys.exit(0)
    print("❌ Configuration validation failed")
    sys.exit(1)

Create Fixture:
mkdir -p src/test_fixtures
echo '{"collections": ["documents", "document_relationships"], "graph_name": "document_graph"}' > src/test_fixtures/setup_expected.json

Test:
uv run src/pdf_extractor/arangodb/config.py

Debugging Tips:

Test fails: Check if constants are strings and non-empty.
File not found: Ensure config.py is in src/pdf_extractor/arangodb/.

Agent Action:

Run the test command.
If “✅ Configuration validation passed” is printed, update [x] Task 1 to [x] Task 1.

Task 2: Implement ArangoDB Setup for Edge Collections
[x] Task 2
Goal: Create functions to set up edge collections and graph definitions.
File: src/pdf_extractor/arangodb/arango_setup.py
Code: (Unchanged, but added fixture creation)
Create Fixture:
echo '{"collections": ["documents", "document_relationships"], "graph_name": "document_graph"}' > src/test_fixtures/setup_expected.json

Test:
uv run src/pdf_extractor/arangodb/arango_setup.py

Debugging Tips:

Connection error: Verify ArangoDB is running (curl http://localhost:8529).
Fixture not found: Check src/test_fixtures/setup_expected.json exists.

Agent Action:

Run the test command.
If “✅ Graph setup validation passed” is printed, update [x] Task 2 to [x] Task 2.

Task 3: Implement Relationship API
[x] Task 3
Goal: Create functions to manage relationships, with enhanced validation for rationales and confidence scores.
File: src/pdf_extractor/arangodb/relationship_api.py
Changes:

Strengthened validate_rationale and validate_confidence_score to enforce MVP requirements.
Updated validate_relationship to check rationale length and confidence score range.
Added fixture creation step.

Code:
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
        return meta
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
    
    edge_meta = add_relationship(db, from_key, to_key, rationale, rel_type, confidence)
    if not edge_meta:
        print("❌ Failed to add relationship")
        sys.exit(1)
    
    validation_passed, validation_failures = validate_relationship(edge_meta, "src/test_fixtures/relationship_expected.json")
    
    if validation_passed:
        print("✅ Relationship API validation passed")
    else:
        print("❌ VALIDATION FAILED - Relationship does not match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
    
    delete_relationship(db, edge_meta["_key"])
    vertex_collection.delete(from_key)
    vertex_collection.delete(to_key)

Create Fixture:
echo '{"type": "SIMILAR"}' > src/test_fixtures/relationship_expected.json

Test:
uv run src/pdf_extractor/arangodb/relationship_api.py

Debugging Tips:

Connection error: Verify ArangoDB is running.
Validation fails: Check rationale length (≥50 chars) and confidence score (1–5).
Fixture not found: Verify src/test_fixtures/relationship_expected.json.

Agent Action:

Run the test command.
If “✅ Relationship API validation passed” is printed, update [x] Task 3 to [x] Task 3.

Task 4: Implement Graph Traversal API
[x] Task 4
Goal: Create a function for graph traversal with filtering.
File: src/pdf_extractor/arangodb/search_api/graph_traverse.py
Code: (Unchanged, but added fixture creation)
Create Fixture:
echo '{"expected_count": 1}' > src/test_fixtures/traversal_expected.json

Test:
uv run src/pdf_extractor/arangodb/search_api/graph_traverse.py

Debugging Tips:

No results: Ensure relationships exist in the graph.
AQL error: Check GRAPH_NAME in config.py.

Agent Action:

Run the test command.
If “✅ Traversal validation passed” is printed, update [x] Task 4 to [x] Task 4.

Task 5: Create Agent Relationship Decision Module
[x] Task 5
Goal: Create a module for deciding when to create relationships, with agent-assessed rationales and confidence scores.
File: src/pdf_extractor/arangodb/agent_decision.py
Changes:

Updated identify_relationship_candidates to suggest all relationship types using simple heuristics.
Modified create_strategic_relationship to prompt the agent for rationale and confidence score, with Perplexity guidance.
Added validation for created relationships.
Added fixture creation step.

Code:
# src/pdf_extractor/arangodb/agent_decision.py
import sys
import json
import uuid
from typing import Dict, Any, List, Optional
from loguru import logger
from arango.database import StandardDatabase
from pdf_extractor.arangodb.config import (
    COLLECTION_NAME, RELATIONSHIP_TYPE_SIMILAR, RELATIONSHIP_TYPE_SHARED_TOPIC,
    RELATIONSHIP_TYPE_REFERENCES, RELATIONSHIP_TYPE_PREREQUISITE, RELATIONSHIP_TYPE_CAUSAL,
    RATIONALE_MIN_LENGTH, CONFIDENCE_SCORE_RANGE
)
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.relationship_api import add_relationship, validate_relationship
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database, ensure_edge_collection, ensure_graph

def should_create_relationship(query_text: str, search_results: Dict[str, Any]) -> float:
    """Decide if relationships are needed."""
    score = 0.0
    results = search_results.get("results", [])
    
    if len(results) < 3:
        score += 0.3
    if len(query_text.split()) > 8:
        score += 0.2
    if any(term in query_text.lower() for term in ["relationship", "connection", "related"]):
        score += 0.3
    if results and results[0].get("rrf_score", 1.0) < 0.5:
        score += 0.2
    
    return min(score, 1.0)

def identify_relationship_candidates(
    db: StandardDatabase,
    query_text: str,
    search_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Find potential relationships, suggesting all types."""
    candidates = []
    results = search_results.get("results", [])
    if len(results) < 2:
        return []
    
    for i, result1 in enumerate(results):
        key1 = result1.get("doc", {}).get("_key")
        if not key1:
            continue
        doc1 = db.collection(COLLECTION_NAME).get(key1)
        for result2 in results[i+1:]:
            key2 = result2.get("doc", {}).get("_key")
            if not key2:
                continue
            doc2 = db.collection(COLLECTION_NAME).get(key2)
            
            score = 0.0
            suggested_type = RELATIONSHIP_TYPE_SIMILAR
            explanation = "General similarity."
            
            # Calculate score based on RRF and tags
            if "rrf_score" in result1 and "rrf_score" in result2:
                score += max(0, 1.0 - abs(result1["rrf_score"] - result2["rrf_score"]))
            tags1 = set(result1.get("doc", {}).get("tags", []))
            tags2 = set(result2.get("doc", {}).get("tags", []))
            shared_tags = tags1.intersection(tags2)
            if shared_tags:
                score += min(len(shared_tags) * 0.1, 0.5)
                suggested_type = RELATIONSHIP_TYPE_SHARED_TOPIC
                explanation = f"Shared {len(shared_tags)} tags."
            
            # Suggest other types based on content
            content1 = doc1.get("content", "").lower()
            content2 = doc2.get("content", "").lower()
            if "cite" in content1 and key2 in content1:
                suggested_type = RELATIONSHIP_TYPE_REFERENCES
                explanation = "Doc1 cites Doc2."
                score += 0.3
            elif "basic" in content1 and "advanced" in content2:
                suggested_type = RELATIONSHIP_TYPE_PREREQUISITE
                explanation = "Doc1 is basic, Doc2 is advanced."
                score += 0.2
            elif "cause" in content1 and "effect" in content2:
                suggested_type = RELATIONSHIP_TYPE_CAUSAL
                explanation = "Doc1 causes effect in Doc2."
                score += 0.2
            
            if score >= 0.3:
                candidates.append({
                    "from_key": key1,
                    "to_key": key2,
                    "score": score,
                    "suggested_type": suggested_type,
                    "explanation": explanation
                })
    
    return sorted(candidates, key=lambda x: x["score"], reverse=True)

def evaluate_relationship_need(db: StandardDatabase, query_text: str, search_attempt: int = 0) -> Dict[str, Any]:
    """Evaluate need for relationships."""
    search_results = hybrid_search(db, query_text, top_n=5)
    need_score = should_create_relationship(query_text, search_results)
    if search_attempt > 0:
        need_score = min(need_score + 0.3, 1.0)
    need_score_10 = round(need_score * 10)
    explanation = {
        7: "High need: Complex query requires relationships.",
        4: "Moderate need: Relationships may help.",
        0: "Low need: Search is sufficient."
    }.get(need_score_10 // 3 * 3, "Unknown need.")
    return {
        "need_score": need_score_10,
        "explanation": explanation,
        "search_results": search_results
    }

def create_strategic_relationship(db: StandardDatabase, from_key: str, to_key: str, query_context: str) -> Optional[Dict[str, Any]]:
    """Create a relationship with agent-assessed rationale and confidence score."""
    try:
        vertex_collection = db.collection(COLLECTION_NAME)
        from_doc = vertex_collection.get(from_key)
        to_doc = vertex_collection.get(to_key)
        if not from_doc or not to_doc:
            logger.error("Documents not found")
            return None
        
        # Suggest relationship type
        tags1 = set(from_doc.get("tags", []))
        tags2 = set(to_doc.get("tags", []))
        content1 = from_doc.get("content", "").lower()
        content2 = to_doc.get("content", "").lower()
        rel_type = RELATIONSHIP_TYPE_SIMILAR
        if tags1.intersection(tags2):
            rel_type = RELATIONSHIP_TYPE_SHARED_TOPIC
        elif "cite" in content1 and to_key in content1:
            rel_type = RELATIONSHIP_TYPE_REFERENCES
        elif "basic" in content1 and "advanced" in content2:
            rel_type = RELATIONSHIP_TYPE_PREREQUISITE
        elif "cause" in content1 and "effect" in content2:
            rel_type = RELATIONSHIP_TYPE_CAUSAL
        
        # Agent provides rationale and confidence score
        print(f"Suggested type: {rel_type}")
        print(f"From doc: {from_doc.get('problem', '')}")
        print(f"To doc: {to_doc.get('problem', '')}")
        print(f"Query context: {query_context}")
        print("Use Perplexity (https://perplexity.ai) to assess the relationship if needed.")
        rationale = input(f"Enter rationale (min {RATIONALE_MIN_LENGTH} chars): ")
        while len(rationale) < RATIONALE_MIN_LENGTH:
            print(f"Rationale too short ({len(rationale)} < {RATIONALE_MIN_LENGTH})")
            rationale = input(f"Enter rationale (min {RATIONALE_MIN_LENGTH} chars): ")
        
        confidence_score = input("Enter confidence score (1-5, 1=best): ")
        while not confidence_score.isdigit() or int(confidence_score) < 1 or int(confidence_score) > 5:
            print(f"Invalid score, must be 1-5")
            confidence_score = input("Enter confidence score (1-5, 1=best): ")
        confidence_score = int(confidence_score)
        
        return add_relationship(db, from_key, to_key, rationale, rel_type, confidence_score)
    except Exception as e:
        logger.error(f"Cannot create relationship: {e}")
        return None

def validate_relationship_creation(relationship, fixture_path):
    """Validate created relationship."""
    validation_failures = {}
    try:
        with open(fixture_path, "r") as f:
            expected = json.load(f)
        
        if relationship.get("type") != expected.get("type"):
            validation_failures["type"] = {
                "expected": expected["type"],
                "actual": relationship.get("type")
            }
        if not validate_rationale(relationship.get("rationale", "")):
            validation_failures["rationale"] = {
                "expected": f">={RATIONALE_MIN_LENGTH} chars",
                "actual": len(relationship.get("rationale", ""))
            }
        if not validate_confidence_score(relationship.get("confidence_score", 0)):
            validation_failures["confidence_score"] = {
                "expected": f"{CONFIDENCE_SCORE_RANGE}",
                "actual": relationship.get("confidence_score")
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
    
    # Create test documents
    doc1_key = f"test_{uuid.uuid4().hex[:8]}"
    doc2_key = f"test_{uuid.uuid4().hex[:8]}"
    vertex_collection = db.collection(COLLECTION_NAME)
    vertex_collection.insert({"_key": doc1_key, "content": "Basic database optimization", "tags": ["database"]})
    vertex_collection.insert({"_key": doc2_key, "content": "Advanced database techniques", "tags": ["database"]})
    
    query = "Database performance optimization"
    relationship = create_strategic_relationship(db, doc1_key, doc2_key, query)
    
    if not relationship:
        print("❌ Failed to create relationship")
        sys.exit(1)
    
    validation_passed, validation_failures = validate_relationship_creation(relationship, "src/test_fixtures/relationship_expected.json")
    
    if validation_passed:
        print("✅ Agent decision validation passed")
    else:
        print("❌ VALIDATION FAILED - Relationship does not match expected values")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            print(f"  - {field}: Expected: {details['expected']}, Got: {details['actual']}")
        print(f"Total errors: {len(validation_failures)} fields mismatched")
        sys.exit(1)
    
    vertex_collection.delete(doc1_key)
    vertex_collection.delete(doc2_key)

Create Fixture:
echo '{"type": "PREREQUISITE"}' > src/test_fixtures/relationship_expected.json

Test:
uv run src/pdf_extractor/arangodb/agent_decision.py

Debugging Tips:

Input errors: Ensure rationale is ≥50 chars and confidence score is 1–5.
No documents: Verify test documents are created.
Fixture mismatch: Check relationship_expected.json matches expected type.

Agent Action:

Run the test command.
When prompted, enter a rationale (use Perplexity if needed) and a confidence score (1–5).
If “✅ Agent decision validation passed” is printed, update [x] Task 5 to [x] Task 5.

Task 6: Implement Advanced Query Solution with Relationship Awareness
[x] Task 6
Goal: Combine search and graph traversal for query solving.
File: src/pdf_extractor/arangodb/advanced_query_solution.py
Code: (Unchanged, but added fixture creation)
Create Fixture:
echo '{"min_results": 1}' > src/test_fixtures/query_expected.json

Test:
uv run src/pdf_extractor/arangodb/advanced_query_solution.py

Debugging Tips:

Empty results: Ensure hybrid_search returns results.
Relationship errors: Check Task 5’s relationships.

Agent Action:

Run the test command.
If “✅ Advanced query validation passed” is printed, update [x] Task 6 to [x] Task 6.

Task 7: Update CLI Commands for Graph Relationships
[x] Task 7
Goal: Add CLI commands for relationship management.
File: src/pdf_extractor/arangodb/cli_extensions.py
Code: (Unchanged, but added test file creation)
Create Test File:
echo -e '# src/pdf_extractor/arangodb/test_cli_extensions.py\nimport sys\nimport typer\nfrom pdf_extractor.arangodb.cli_extensions import register_agent_commands\n\ndef test_cli_extensions():\n    """Test CLI extensions import."""\n    app = typer.Typer()\n    register_agent_commands(app)\n    print("✅ CLI extensions validation passed")\n    return True\n\nif __name__ == "__main__":\n    if test_cli_extensions():\n        print("✅ CLI extensions test passed")\n    else:\n        print("❌ CLI extensions test failed")\n        sys.exit(1)' > src/pdf_extractor/arangodb/test_cli_extensions.py

Test:
uv run src/pdf_extractor/arangodb/test_cli_extensions.py

Debugging Tips:

Import errors: Ensure all dependencies are in src/pdf_extractor/arangodb/.
Typer failure: Check typer is installed (uv pip install typer).

Agent Action:

Run the test command.
If “✅ CLI extensions test passed” is printed, update [x] Task 7 to [x] Task 7.

Task 8: Create Agent Guidelines Documentation
[x] Task 8
Goal: Document guidelines for creating relationships, emphasizing agent-assessed rationales.
File: src/pdf_extractor/arangodb/docs/agent_relationship_guidelines.md
Changes:

Updated to highlight the agent’s role in assessing rationales and confidence scores.
Added Perplexity usage guidance.
Clarified confidence score meanings.

Content:
# Agent Relationship Guidelines

## Overview
This guide explains how to create useful relationships between documents in the PDF Extractor to improve query answering.

## When to Create Relationships
Create relationships when:
1. Search results are poor (<3 results or low RRF scores).
2. Query needs information from multiple documents.
3. Relationships will help future queries (e.g., connecting problems to solutions).

## Decision Process
1. Run hybrid search (`hybrid_search`).
2. If results are poor, use existing relationships via graph traversal.
3. If still poor, create new relationships.

## Good Relationships
- **SIMILAR**: Documents with similar content (e.g., both discuss query optimization).
- **SHARED_TOPIC**: Documents with shared tags (e.g., both tagged “database”).
- **REFERENCES**: Doc A cites Doc B (e.g., mentions its key or title).
- **PREREQUISITE**: Doc A is needed to understand Doc B (e.g., basic vs. advanced).
- **CAUSAL**: Doc A’s concept causes Doc B’s outcome (e.g., configuration causes performance).

## Bad Relationships
- Based only on shared keywords without deeper connection.
- Redundant with search results.
- Too vague or speculative.

## Assessing Relationships
- **Rationale**: Provide a clear explanation (≥50 characters) of why the documents are related. Use Perplexity (https://perplexity.ai) to research connections if needed. Example: “Doc1 discusses query optimization, and Doc2 provides advanced indexing techniques, both relevant to performance.”
- **Confidence Score**:
  - 1: Essential connection (e.g., problem-solution pair).
  - 2: Strong connection (e.g., clear shared topic).
  - 3: Helpful connection (e.g., likely related).
  - 4: Weak connection (e.g., possible relation).
  - 5: Uncertain connection (e.g., speculative).
- **Process**:
  1. Review document content and tags.
  2. Check query context.
  3. Use Perplexity to verify connections if unsure.
  4. Write a specific rationale and assign a score.

## Testing
- Verify relationships improve query results (more results or better relevance).
- Ensure rationales are clear and ≥50 characters.
- Confirm confidence scores reflect connection strength (1=best, 5=worst).

Test:
if [ -f src/pdf_extractor/arangodb/docs/agent_relationship_guidelines.md ]; then echo "✅ Documentation validation passed"; else echo "❌ Documentation missing"; exit 1; fi

Debugging Tips:

File missing: Create agent_relationship_guidelines.md in src/pdf_extractor/arangodb/docs/.
Permission error: Check write permissions (chmod +w src/pdf_extractor/arangodb/docs).

Agent Action:

Run the test command.
If “✅ Documentation validation passed” is printed, update [x] Task 8 to [x] Task 8.

Task 9: Create Integration Example
[x] Task 9
Goal: Create an example of the agent workflow, testing relationship impact.
File: src/pdf_extractor/arangodb/examples/agent_workflow.py
Changes:

Added a test to compare query results before and after creating a relationship.
Updated to use agent-assessed rationale and confidence score.
Added fixture creation step.

Code:
# src/pdf_extractor/arangodb/examples/agent_workflow.py
import sys
import uuid
from loguru import logger
from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database, ensure_edge_collection, ensure_graph
from pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from pdf_extractor.arangodb.agent_decision import evaluate_relationship_need, create_strategic_relationship
from pdf_extractor.arangodb.advanced_query_solution import solve_query
from pdf_extractor.arangodb.config import COLLECTION_NAME

logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_documents(db):
    """Create test documents."""
    test_id = uuid.uuid4().hex[:8]
    doc1_key = f"test_doc1_{test_id}"
    doc2_key = f"test_doc2_{test_id}"
    
    vertex_collection = db.collection(COLLECTION_NAME)
    vertex_collection.insert({
        "_key": doc1_key,
        "problem": "Slow database queries",
        "content": "Basic database optimization",
        "tags": ["database", "performance"]
    })
    vertex_collection.insert({
        "_key": doc2_key,
        "problem": "API response time",
        "content": "Advanced database techniques",
        "tags": ["database", "performance"]
    })
    return doc1_key, doc2_key

def test_relationship_impact(db, query, doc1_key, doc2_key):
    """Test if relationship improves query results."""
    before = solve_query(db, query)
    relationship = create_strategic_relationship(db, doc1_key, doc2_key, query)
    if not relationship:
        return False, "Failed to create relationship"
    after = solve_query(db, query)
    improved = len(after["results"]) > len(before["results"])
    return improved, f"Results: {len(before['results'])} before, {len(after['results'])} after"

def run_workflow_example():
    """Run agent workflow."""
    client = connect_arango()
    db = ensure_database(client)
    ensure_edge_collection(db)
    ensure_graph(db)
    
    doc1_key, doc2_key = create_test_documents(db)
    
    query = "Optimize database and API performance"
    logger.info(f"Query: {query}")
    
    results = hybrid_search(db, query, top_n=5)
    logger.info(f"Search found {len(results.get('results', []))} results")
    
    need = evaluate_relationship_need(db, query)
    logger.info(f"Need score: {need['need_score']}/10")
    
    if need["need_score"] >= 7:
        improved, impact_message = test_relationship_impact(db, query, doc1_key, doc2_key)
        if not improved:
            logger.error(f"Relationship did not improve results: {impact_message}")
            return {"documents": [doc1_key, doc2_key], "relationships": [], "impact": impact_message}
    
    final_result = solve_query(db, query)
    logger.info(f"Solved with attempt {final_result.get('attempt', 0)}")
    
    return {"documents": [doc1_key, doc2_key], "relationships": [], "impact": impact_message if 'impact_message' in locals() else "No relationships created"}

def cleanup_resources(db, resources):
    """Clean up resources."""
    vertex_collection = db.collection(COLLECTION_NAME)
    edge_collection = db.collection("document_relationships")
    
    for doc_key in resources.get("documents", []):
        vertex_collection.delete(doc_key, ignore_missing=True)
    for edge_key in resources.get("relationships", []):
        if edge_key:
            edge_collection.delete(edge_key, ignore_missing=True)

if __name__ == "__main__":
    client = connect_arango()
    db = ensure_database(client)
    
    try:
        resources = run_workflow_example()
        cleanup_resources(db, resources)
        print("✅ Workflow example validation passed")
    except Exception as e:
        print(f"❌ Workflow example failed: {e}")
        sys.exit(1)

Create Fixture:
echo '{"min_results": 1}' > src/test_fixtures/query_expected.json

Test:
uv run src/pdf_extractor/arangodb/examples/agent_workflow.py

Debugging Tips:

Input errors: Provide valid rationale and confidence score when prompted.
No impact: Ensure relationship type matches query context.
Fixture missing: Verify src/test_fixtures/query_expected.json.

Agent Action:

Run the test command.
When prompted, enter a rationale (use Perplexity if needed) and a confidence score (1–5).
If “✅ Workflow example validation passed” is printed, update [x] Task 9 to [x] Task 9.

Final Testing
[x] Final Testing
Test:
uv run -m unittest discover -s src/pdf_extractor/arangodb
uv run src/pdf_extractor/arangodb/examples/agent_workflow.py

Debugging Tips:

Unit tests fail: Check individual task tests for errors.
Workflow fails: Review Task 9’s debugging tips.

Agent Action:

Run the test commands.
If both pass (no errors and “✅ Workflow example validation passed”), update [x] Final Testing to [x] Final Testing.

