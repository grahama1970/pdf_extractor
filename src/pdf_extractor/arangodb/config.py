# src/pdf_extractor/arangodb/config.py
import os

# ArangoDB Connection Settings
ARANGO_HOST = os.environ.get("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.environ.get("ARANGO_USER", "root")
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
ARANGO_DB_NAME = os.environ.get("ARANGO_DB_NAME", "pdf_extractor")

# Collection & View Names
COLLECTION_NAME = "documents"
EDGE_COLLECTION_NAME = "relationships"
VIEW_NAME = "document_view"
GRAPH_NAME = "knowledge_graph"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSIONS = 1536

# Search Configuration
SEARCH_FIELDS = ["problem", "solution", "context", "title", "tags"]
ALL_DATA_FIELDS_PREVIEW = ["_key", "problem", "solution", "tags", "title", "context"]

# Analyzer Configuration
TEXT_ANALYZER = "text_en"
TAG_ANALYZER = "text_en"
