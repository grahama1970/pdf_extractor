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
ARANGO_PASSWORD = "openSesame"

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

# Embedding settings
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536

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
