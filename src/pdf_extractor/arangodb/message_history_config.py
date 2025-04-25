# src/pdf_extractor/arangodb/message_history_config.py
"""
Configuration module for conversation history storage in ArangoDB.

This module defines constants and schema definitions for storing
conversation messages between users and the agent in ArangoDB.
"""
import sys
from typing import Dict, Any, List, Tuple

# Collection and graph names
MESSAGE_COLLECTION_NAME = "claude_message_history"
MESSAGE_EDGE_COLLECTION_NAME = "message_relationships"
MESSAGE_GRAPH_NAME = "conversation_graph"

# Message types
MESSAGE_TYPE_USER = "USER"
MESSAGE_TYPE_AGENT = "AGENT"
MESSAGE_TYPE_SYSTEM = "SYSTEM"

# Message validation settings
MESSAGE_MAX_LENGTH = 100000  # Max characters in a message
MESSAGE_MIN_LENGTH = 1  # Min characters in a message
CONVERSATION_ID_LENGTH = 36  # UUID length

# Message schema properties
REQUIRED_MESSAGE_FIELDS = [
    "conversation_id",
    "message_type",
    "content",
    "timestamp"
]

# Message relationship types
RELATIONSHIP_TYPE_NEXT = "NEXT_MESSAGE"  # Sequential relationship
RELATIONSHIP_TYPE_REFERS_TO = "REFERS_TO_DOCUMENT"  # Reference to a document
RELATIONSHIP_TYPE_CLARIFIES = "CLARIFIES"  # Message clarifies previous message
RELATIONSHIP_TYPE_ANSWERS = "ANSWERS"  # Message answers previous question

# Index configuration
MESSAGE_INDEXES = [
    {
        "fields": ["conversation_id"],
        "type": "persistent",
        "unique": False
    },
    {
        "fields": ["timestamp"],
        "type": "persistent",
        "unique": False
    },
    {
        "fields": ["message_type"],
        "type": "persistent",
        "unique": False
    }
]

# Search fields for message content
MESSAGE_SEARCH_FIELDS = ["content", "metadata.tags"]
MESSAGE_TEXT_ANALYZER = "text_en"

def validate_message_config():
    """Validate the message history configuration."""
    # Validate collection names are not empty
    if not MESSAGE_COLLECTION_NAME or not MESSAGE_EDGE_COLLECTION_NAME or not MESSAGE_GRAPH_NAME:
        return False, "Collection or graph names cannot be empty"
    
    # Validate message length constraints
    if MESSAGE_MIN_LENGTH < 1:
        return False, "Minimum message length must be at least 1"
    
    if MESSAGE_MAX_LENGTH < MESSAGE_MIN_LENGTH:
        return False, "Maximum message length must be greater than minimum length"
    
    # Validate required fields
    if not REQUIRED_MESSAGE_FIELDS or len(REQUIRED_MESSAGE_FIELDS) == 0:
        return False, "Required message fields cannot be empty"
    
    return True, "Message history configuration is valid"

if __name__ == "__main__":
    valid, message = validate_message_config()
    if valid:
        print("✅ Message history configuration validation passed")
        sys.exit(0)
    else:
        print(f"❌ Message history configuration validation failed: {message}")
        sys.exit(1)
