# src/pdf_extractor/arangodb/message_history_test_data.py
"""
Tool for generating and inserting test message history data.

This module provides utilities to generate realistic test conversation data
and insert it into the message history collection.
"""
import sys
import json
import uuid
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger

from arango.database import StandardDatabase

from pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from pdf_extractor.arangodb.message_history_setup import initialize_message_history
from pdf_extractor.arangodb.message_history_api import (
    add_message, 
    get_conversation_messages,
    delete_conversation
)
from pdf_extractor.arangodb.message_history_config import (
    MESSAGE_TYPE_USER,
    MESSAGE_TYPE_AGENT,
    MESSAGE_TYPE_SYSTEM
)

# Sample conversation topics for generating test data
TOPICS = [
    "database optimization",
    "machine learning algorithms",
    "web development frameworks",
    "cloud infrastructure",
    "agile methodologies",
    "software testing",
    "cybersecurity best practices",
    "data visualization",
    "natural language processing",
    "mobile app development"
]

# Sample user messages for each topic
USER_MESSAGES = {
    "database optimization": [
        "How can I optimize my PostgreSQL database for better performance?",
        "What indexes should I create for my query workload?",
        "Are there any tools for identifying slow queries in my database?",
        "How do I implement database partitioning for large tables?",
        "What's the best way to optimize join operations in SQL?"
    ],
    "machine learning algorithms": [
        "What's the difference between supervised and unsupervised learning?",
        "How do I choose the right algorithm for my classification problem?",
        "Can you explain how neural networks work?",
        "What are some techniques for handling imbalanced datasets?",
        "How do I evaluate the performance of my machine learning model?"
    ],
    "web development frameworks": [
        "What are the pros and cons of React vs Angular?",
        "How do I set up a new project with Next.js?",
        "What's the best backend framework to use with React?",
        "How do I implement authentication in my web app?",
        "What are some good practices for API design?"
    ],
    "cloud infrastructure": [
        "How do I set up auto-scaling for my application on AWS?",
        "What's the difference between containers and virtual machines?",
        "How can I optimize my cloud infrastructure costs?",
        "What are the best practices for cloud security?",
        "How do I implement a CI/CD pipeline in the cloud?"
    ],
    "agile methodologies": [
        "What's the difference between Scrum and Kanban?",
        "How do I implement Agile in a remote team?",
        "What are some good tools for Agile project management?",
        "How do we conduct effective sprint retrospectives?",
        "What metrics should we track in our Agile process?"
    ],
    "software testing": [
        "What's the difference between unit and integration testing?",
        "How do I implement test-driven development?",
        "What are some good testing frameworks for Python?",
        "How do I write effective test cases?",
        "What's the best approach for testing asynchronous code?"
    ],
    "cybersecurity best practices": [
        "How do I implement proper password hashing in my application?",
        "What are some common security vulnerabilities to watch out for?",
        "How do I set up two-factor authentication?",
        "What are the best practices for securing API endpoints?",
        "How do I conduct a security audit of my application?"
    ],
    "data visualization": [
        "What are the best tools for creating interactive dashboards?",
        "How do I choose the right chart type for my data?",
        "What are some principles of effective data visualization?",
        "How do I create accessible data visualizations?",
        "What's the best way to visualize time-series data?"
    ],
    "natural language processing": [
        "How do I implement sentiment analysis for customer reviews?",
        "What are the recent advances in NLP?",
        "How do I fine-tune a pre-trained language model?",
        "What are some good libraries for text processing in Python?",
        "How do I build a simple chatbot using NLP?"
    ],
    "mobile app development": [
        "What are the differences between native and cross-platform development?",
        "How do I implement push notifications in my mobile app?",
        "What are some best practices for mobile UI design?",
        "How do I optimize my mobile app for performance?",
        "What are some good frameworks for cross-platform development?"
    ]
}

# Sample agent responses for each topic
AGENT_RESPONSES = {
    "database optimization": [
        "To optimize your PostgreSQL database, you could start by analyzing slow queries using EXPLAIN ANALYZE, creating appropriate indexes, optimizing your schema design, and tuning configuration parameters like shared_buffers and work_mem.",
        "For your query workload, consider creating indexes on columns frequently used in WHERE clauses, JOIN conditions, and ORDER BY statements. Use EXPLAIN ANALYZE to verify if your indexes are being used effectively.",
        "Yes, PostgreSQL offers several tools for identifying slow queries: pg_stat_statements extension tracks query performance metrics, auto_explain module logs execution plans for slow queries, and pgBadger analyzes log files to generate performance reports.",
        "To implement database partitioning in PostgreSQL, you can use declarative partitioning (available since version 10) by creating a parent table with a partitioning strategy (RANGE, LIST, or HASH) and then creating child tables for each partition.",
        "To optimize join operations in SQL, ensure you have appropriate indexes on join columns, order your joins to process the smallest dataset first, use JOIN instead of subqueries when possible, and consider denormalizing data if joins are still slow."
    ],
    "machine learning algorithms": [
        "The key difference is that supervised learning uses labeled data to train models to make predictions, while unsupervised learning works with unlabeled data to find patterns and relationships. Semi-supervised learning uses both labeled and unlabeled data.",
        "When choosing a classification algorithm, consider your data size, features, interpretability needs, and accuracy requirements. Decision trees work well for categorical features, random forests handle high-dimensional data, and neural networks excel with large datasets.",
        "Neural networks consist of layers of interconnected nodes (neurons) that process input data through weighted connections. Each neuron applies an activation function to weighted inputs, and the network learns by adjusting weights to minimize prediction error through backpropagation.",
        "For imbalanced datasets, you can use sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic examples, apply class weights to penalize misclassification of minority classes, or use ensemble methods like RUSBoost.",
        "To evaluate your machine learning model, use metrics appropriate for your problem: accuracy for balanced classes, precision/recall/F1-score for imbalanced data, AUC-ROC for ranking performance, and cross-validation to assess generalization ability."
    ],
    # Additional responses for other topics would be added here
    "web development frameworks": [
        "React and Angular are both powerful frameworks but differ in philosophy. React is a library focused on UI components with a unidirectional data flow, while Angular is a comprehensive framework with two-way data binding. React has a smaller learning curve and more flexibility, while Angular provides more structure and built-in solutions.",
        "To set up a Next.js project, first install Node.js, then run 'npx create-next-app my-app'. This creates a new project with the basic structure. Navigate to the directory and run 'npm run dev' to start the development server. You can then begin creating pages in the 'pages' directory.",
        "For a React frontend, popular backend frameworks include Express.js (Node.js), Django REST (Python), Ruby on Rails, and Spring Boot (Java). Express is commonly used due to the JavaScript consistency, but the best choice depends on your team's expertise and specific requirements.",
        "To implement authentication, you can use solutions like Auth0, Firebase Authentication, or roll your own with JWT tokens. The basic flow involves user registration, secure password storage, login endpoint that returns tokens, and protected routes that verify these tokens.",
        "Good API design practices include using RESTful principles, versioning your APIs, implementing proper error handling with meaningful status codes, providing comprehensive documentation, ensuring security with authentication/authorization, and designing with pagination for large data sets."
    ]
}

def generate_test_conversation(
    topic: Optional[str] = None,
    num_exchanges: int = 3,
    include_system_message: bool = True,
    start_time: Optional[datetime] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a test conversation with user and agent messages.
    
    Args:
        topic: Optional topic for the conversation
        num_exchanges: Number of user-agent exchanges to generate
        include_system_message: Whether to include a system message
        start_time: Optional start time for the conversation
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: conversation_id and list of message objects
    """
    # Generate a conversation ID
    conversation_id = str(uuid.uuid4())
    
    # Select a random topic if not specified
    if not topic:
        topic = random.choice(TOPICS)
    
    # Set start time if not specified
    if not start_time:
        # Random time within the last 30 days
        days_ago = random.randint(0, 30)
        start_time = datetime.now(timezone.utc) - timedelta(days=days_ago)
    
    # List to store generated messages
    messages = []
    
    # Add a system message if requested
    if include_system_message:
        system_message = {
            "conversation_id": conversation_id,
            "message_type": MESSAGE_TYPE_SYSTEM,
            "content": f"Conversation started on topic: {topic}",
            "timestamp": start_time.isoformat(),
            "metadata": {
                "topic": topic,
                "tags": [topic.split()[0], "test_data"]
            }
        }
        messages.append(system_message)
        
        # Advance time for next message
        start_time += timedelta(seconds=random.randint(5, 30))
    
    # Generate user-agent exchanges
    for i in range(num_exchanges):
        # Select random user message for this topic
        if topic in USER_MESSAGES and USER_MESSAGES[topic]:
            user_content = random.choice(USER_MESSAGES[topic])
        else:
            user_content = f"Question about {topic} (exchange {i+1})"
        
        # Create user message
        user_message = {
            "conversation_id": conversation_id,
            "message_type": MESSAGE_TYPE_USER,
            "content": user_content,
            "timestamp": start_time.isoformat(),
            "metadata": {
                "user_id": f"test_user_{random.randint(1, 5)}",
                "topic": topic,
                "tags": [topic.split()[0], "test_data"]
            }
        }
        messages.append(user_message)
        
        # Advance time for agent response
        start_time += timedelta(seconds=random.randint(20, 60))
        
        # Select random agent response for this topic
        if topic in AGENT_RESPONSES and AGENT_RESPONSES[topic]:
            agent_content = random.choice(AGENT_RESPONSES[topic])
        else:
            agent_content = f"Response about {topic} (exchange {i+1})"
        
        # Create agent message
        agent_message = {
            "conversation_id": conversation_id,
            "message_type": MESSAGE_TYPE_AGENT,
            "content": agent_content,
            "timestamp": start_time.isoformat(),
            "metadata": {
                "agent_id": "claude",
                "topic": topic,
                "tags": [topic.split()[0], "test_data"]
            }
        }
        messages.append(agent_message)
        
        # Advance time for next exchange
        start_time += timedelta(minutes=random.randint(1, 5))
    
    return conversation_id, messages

def insert_test_conversation(
    db: StandardDatabase,
    conversation_id: Optional[str] = None,
    topic: Optional[str] = None,
    num_exchanges: int = 3,
    include_system_message: bool = True
) -> str:
    """
    Generate and insert a test conversation into the database.
    
    Args:
        db: ArangoDB database handle
        conversation_id: Optional conversation ID
        topic: Optional topic for the conversation
        num_exchanges: Number of user-agent exchanges to generate
        include_system_message: Whether to include a system message
        
    Returns:
        str: The conversation ID
    """
    # Generate test conversation
    conv_id, messages = generate_test_conversation(
        topic=topic,
        num_exchanges=num_exchanges,
        include_system_message=include_system_message
    )
    
    # Use provided conversation ID if specified
    if conversation_id:
        conv_id = conversation_id
        for msg in messages:
            msg["conversation_id"] = conv_id
    
    # Insert messages in sequence
    previous_key = None
    for msg in messages:
        result = add_message(
            db=db,
            conversation_id=msg["conversation_id"],
            message_type=msg["message_type"],
            content=msg["content"],
            metadata=msg["metadata"],
            timestamp=msg["timestamp"],
            previous_message_key=previous_key
        )
        
        if result:
            previous_key = result["_key"]
        else:
            logger.warning(f"Failed to insert message: {msg}")
    
    logger.info(f"Inserted {len(messages)} messages for conversation: {conv_id}")
    return conv_id

def bulk_insert_test_conversations(
    db: StandardDatabase,
    num_conversations: int = 5,
    exchanges_per_conversation: int = 3,
    include_system_messages: bool = True
) -> List[str]:
    """
    Generate and insert multiple test conversations.
    
    Args:
        db: ArangoDB database handle
        num_conversations: Number of conversations to generate
        exchanges_per_conversation: Number of exchanges per conversation
        include_system_messages: Whether to include system messages
        
    Returns:
        List[str]: List of conversation IDs
    """
    conversation_ids = []
    
    for i in range(num_conversations):
        # Select random topic
        topic = random.choice(TOPICS)
        
        # Random number of exchanges (variation)
        exchanges = random.randint(
            max(1, exchanges_per_conversation - 1),
            exchanges_per_conversation + 1
        )
        
        # Insert the conversation
        conv_id = insert_test_conversation(
            db=db,
            topic=topic,
            num_exchanges=exchanges,
            include_system_message=include_system_messages
        )
        
        conversation_ids.append(conv_id)
        logger.info(f"Created test conversation {i+1}/{num_conversations}: {conv_id}")
    
    return conversation_ids

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and insert test message history data")
    parser.add_argument(
        "--conversations", 
        type=int, 
        default=5,
        help="Number of conversations to generate"
    )
    parser.add_argument(
        "--exchanges", 
        type=int, 
        default=3,
        help="Number of exchanges per conversation"
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not include system messages"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up existing test data first"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Connect to ArangoDB and ensure database exists
    client = connect_arango()
    if not client:
        print("❌ Failed to connect to ArangoDB")
        sys.exit(1)
    
    db = ensure_database(client)
    if not db:
        print("❌ Failed to ensure database exists")
        sys.exit(1)
    
    # Initialize message history collections if needed
    initialize_message_history(db)
    
    # Clean up existing test data if requested
    if args.clean:
        logger.info("Cleaning up existing test data...")
        
        # Query test conversations
        aql = """
        FOR msg IN claude_message_history
        FILTER msg.metadata.tags ANY == "test_data"
        COLLECT conversation_id = msg.conversation_id
        RETURN conversation_id
        """
        
        cursor = db.aql.execute(aql)
        test_conversation_ids = list(cursor)
        
        # Delete each test conversation
        for conv_id in test_conversation_ids:
            delete_conversation(db, conv_id)
        
        logger.info(f"Cleaned up {len(test_conversation_ids)} test conversations")
    
    # Generate and insert test conversations
    conversation_ids = bulk_insert_test_conversations(
        db=db,
        num_conversations=args.conversations,
        exchanges_per_conversation=args.exchanges,
        include_system_messages=not args.no_system
    )
    
    print(f"✅ Successfully generated {len(conversation_ids)} test conversations")
    print(f"Conversation IDs: {conversation_ids}")
    sys.exit(0)
