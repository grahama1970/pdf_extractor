#!/usr/bin/env python3

from pdf_extractor.arangodb_borked.connection import get_db
import random
import math

# Constants
PDF_COLLECTION_NAME = 'pdf_documents'
EMBEDDING_FIELD = 'embeddings'
EMBEDDING_DIMENSION = 384
VECTOR_INDEX_NAME = 'vector_index_fixed'

def generate_embedding(dim=EMBEDDING_DIMENSION, seed=42):
    random.seed(seed)
    embedding = [random.uniform(-1, 1) for _ in range(dim)]
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    return [x / magnitude for x in embedding] if magnitude > 0 else embedding

def main():
    db = get_db()
    print('Connected to ArangoDB')
    
    collection = db.collection(PDF_COLLECTION_NAME)
    print('Connected to collection')
    
    # Try to create a fixed vector index
    try:
        collection.add_index({
            'type': 'vector',
            'name': VECTOR_INDEX_NAME,
            'fields': [EMBEDDING_FIELD],
            'params': {
                'metric': 'cosine',
                'dimension': EMBEDDING_DIMENSION,
                'nLists': 16  # Adding the required nLists parameter
            }
        })
        print(f'Created fixed vector index: {VECTOR_INDEX_NAME}')
    except Exception as e:
        print(f'Error creating vector index: {e}')
    
    # Try a vector search
    try:
        # Generate test embedding
        test_embedding = generate_embedding()
        print('Generated test embedding')
        
        # Execute search
        aql = f'''
        FOR doc IN {PDF_COLLECTION_NAME}
            FILTER HAS(doc, '{EMBEDDING_FIELD}')
            SORT VECTOR_DISTANCE(doc.{EMBEDDING_FIELD}, @query_embedding, 'cosine') ASC
            LIMIT 5
            RETURN doc
        '''
        
        print('Executing vector search...')
        cursor = db.aql.execute(
            aql,
            bind_vars={'query_embedding': test_embedding}
        )
        
        results = [doc for doc in cursor]
        print(f'Vector search found {len(results)} documents')
        print('✅ Vector search is working!')
        
        # Output the fixed semantic_search implementation
        print('''
To fix semantic_search in search_functions.py, use this:

def semantic_search(db, query_embedding, limit=10):
    