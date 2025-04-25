#!/usr/bin/env python3
# tests/validate_all_search_functions.py

import sys
import json
import os
from loguru import logger

# Set up the Python path correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_extractor.arangodb.arango_setup import connect_arango, ensure_database
from src.pdf_extractor.arangodb.config import COLLECTION_NAME as DOC_COLLECTION_NAME
from src.pdf_extractor.arangodb.message_history_config import MESSAGE_COLLECTION_NAME

# Import search functions
from src.pdf_extractor.arangodb.search_api.tag_search import tag_search
from src.pdf_extractor.arangodb.search_api.glossary import glossary_search
from src.pdf_extractor.arangodb.search_api.hybrid import hybrid_search
from src.pdf_extractor.arangodb.search_api.semantic import semantic_search
from src.pdf_extractor.arangodb.search_api.bm25 import bm25_search
from src.pdf_extractor.arangodb.search_api.keyword import search_keyword

def test_connection():
    """Test connection to ArangoDB and print version."""
    try:
        client = connect_arango()
        db = ensure_database(client)
        print(f"✅ Connected to ArangoDB {db.version()}")
        return db
    except Exception as e:
        print(f"❌ Connection error: {e}")
        sys.exit(1)

def test_tag_search(db):
    """Test tag search functionality."""
    print("\n--- Testing Tag Search ---")
    try:
        # Sample tags to search for
        tags = ["python", "json", "pdf"] 
        results = tag_search(db, tags=tags, limit=5)
        
        print(f"Query: tags={tags}")
        print(f"Results: {len(results.get('results', []))} documents found")
        print(f"Total: {results.get('total', 0)} documents matching tags")
        
        if results.get('results'):
            first_result = results['results'][0]
            print("\nFirst result preview:")
            if 'doc' in first_result:
                print(f"  Document Key: {first_result['doc'].get('_key')}")
                if 'tags' in first_result['doc']:
                    print(f"  Tags: {first_result['doc']['tags']}")
                if 'title' in first_result['doc']:
                    print(f"  Title: {first_result['doc']['title']}")
            
            return True
        else:
            if results.get('error'):
                print(f"Error: {results.get('error')}")
            print("No results found or error occurred.")
            return False
    except Exception as e:
        print(f"❌ Tag search error: {e}")
        return False

def test_glossary_search(db):
    """Test glossary search functionality."""
    print("\n--- Testing Glossary Search ---")
    try:
        # Sample terms to search for
        terms = ["error", "python", "function"]
        results = glossary_search(db, terms=terms, limit=5)
        
        print(f"Query: terms={terms}")
        print(f"Results: {len(results.get('results', []))} documents found")
        print(f"Count: {results.get('count', 0)} documents matching terms")
        
        if results.get('results'):
            first_result = results['results'][0]
            print("\nFirst result preview:")
            if 'doc' in first_result:
                print(f"  Document Key: {first_result['doc'].get('_key')}")
                if 'title' in first_result['doc']:
                    print(f"  Title: {first_result['doc']['title']}")
                # Print first 100 chars of content
                if 'content' in first_result['doc']:
                    content = first_result['doc']['content']
                    print(f"  Content (first 100 chars): {content[:100]}...")
            
            return True
        else:
            if results.get('error'):
                print(f"Error: {results.get('error')}")
            print("No results found or error occurred.")
            return False
    except Exception as e:
        print(f"❌ Glossary search error: {e}")
        return False

def test_hybrid_search(db):
    """Test hybrid search functionality."""
    print("\n--- Testing Hybrid Search ---")
    try:
        # Sample query
        query_text = "pdf extraction validation error"
        results = hybrid_search(db, query_text=query_text, top_n=5)
        
        print(f"Query: '{query_text}'")
        print(f"Results: {len(results.get('results', []))} documents found")
        print(f"Total: {results.get('total', 0)} documents matching query")
        
        if results.get('results'):
            first_result = results['results'][0]
            print("\nFirst result preview:")
            print(f"  Score: {first_result.get('score')}")
            if 'doc' in first_result:
                print(f"  Document Key: {first_result['doc'].get('_key')}")
                if 'title' in first_result['doc']:
                    print(f"  Title: {first_result['doc']['title']}")
                # Print first 100 chars of content
                if 'content' in first_result['doc']:
                    content = first_result['doc']['content']
                    print(f"  Content (first 100 chars): {content[:100]}...")
            
            return True
        else:
            if results.get('error'):
                print(f"Error: {results.get('error')}")
            print("No results found or error occurred.")
            return False
    except Exception as e:
        print(f"❌ Hybrid search error: {e}")
        return False

def test_semantic_search(db):
    """Test semantic search functionality."""
    print("\n--- Testing Semantic Search ---")
    try:
        # Sample query
        query = "How to validate PDF extraction results"
        results = semantic_search(db, query=query, top_n=5)
        
        print(f"Query: '{query}'")
        print(f"Results: {len(results.get('results', []))} documents found")
        
        if results.get('results'):
            first_result = results['results'][0]
            print("\nFirst result preview:")
            print(f"  Similarity Score: {first_result.get('similarity_score')}")
            if 'doc' in first_result:
                print(f"  Document Key: {first_result['doc'].get('_key')}")
                if 'title' in first_result['doc']:
                    print(f"  Title: {first_result['doc']['title']}")
                # Print first 100 chars of content
                if 'content' in first_result['doc']:
                    content = first_result['doc']['content']
                    print(f"  Content (first 100 chars): {content[:100]}...")
            
            return True
        else:
            if results.get('error'):
                print(f"Error: {results.get('error')}")
            print("No results found or error occurred.")
            return False
    except Exception as e:
        print(f"❌ Semantic search error: {e}")
        return False

def test_keyword_search(db):
    """Test keyword search functionality."""
    print("\n--- Testing Keyword Search ---")
    try:
        # Sample search term
        search_term = "python validation pdf"
        results = search_keyword(db, search_term=search_term, top_n=5)
        
        print(f"Query: search_term='{search_term}'")
        print(f"Results: {len(results.get('results', []))} documents found")
        
        if results.get('results'):
            first_result = results['results'][0]
            print("\nFirst result preview:")
            if 'score' in first_result:
                print(f"  Score: {first_result.get('score')}")
            if 'doc' in first_result:
                print(f"  Document Key: {first_result['doc'].get('_key')}")
                if 'title' in first_result['doc']:
                    print(f"  Title: {first_result['doc']['title']}")
            
            return True
        else:
            if results.get('error'):
                print(f"Error: {results.get('error')}")
            print("No results found or error occurred.")
            return False
    except Exception as e:
        print(f"❌ Keyword search error: {e}")
        return False

def run_all_tests():
    """Run all search function tests."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("=== ArangoDB Search Functions Validation ===")
    print("Testing connection to ArangoDB...")
    
    # Test connection
    db = test_connection()
    
    # Run all tests
    tests = [
        ("Tag Search", test_tag_search),
        ("Glossary Search", test_glossary_search),
        ("Hybrid Search", test_hybrid_search),
        ("Semantic Search", test_semantic_search),
        ("Keyword Search", test_keyword_search)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func(db)
    
    # Print summary
    print("\n=== Search Functions Validation Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ ALL SEARCH FUNCTIONS VALIDATED SUCCESSFULLY")
        return 0
    else:
        print("\n❌ SOME SEARCH FUNCTIONS FAILED VALIDATION")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
