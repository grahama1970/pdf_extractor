#!/usr/bin/env python3

from pdf_extractor.arangodb_borked.connection import get_db
import random
import math
import time

# Constants
PDF_COLLECTION_NAME = 'pdf_documents'
EMBEDDING_FIELD = 'embeddings'
EMBEDDING_DIMENSION = 384
VECTOR_INDEX_NAME = f'vector_index_{int(time.time())}'

def generate_embedding(dim=EMBEDDING_DIMENSION, seed=None):
    if seed:
        random.seed(seed)
    embedding = [random.uniform(-1, 1) for _ in range(dim)]
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    return [x / magnitude for x in embedding] if magnitude > 0 else embedding

def main():
    db = get_db()
    print('Connected to ArangoDB')
    
    collection = db.collection(PDF_COLLECTION_NAME)
    print('Connected to collection')
    
    # First, insert some test documents with embeddings
    test_docs = []
    for i in range(10):
        doc = {
            'type': 'test',
            'text': f'Test document {i}',
            'page': i,
            'file_path': 'test.pdf',
            EMBEDDING_FIELD: generate_embedding(seed=i)
        }
        test_docs.append(doc)
    
    print(f'Inserting {len(test_docs)} test documents with embeddings')
    result = collection.insert_many(test_docs)
    print(f'Inserted test documents: {result}')
    
    # Try to create a vector index
    try:
        collection.add_index({
            'type': 'vector',
            'name': VECTOR_INDEX_NAME,
            'fields': [EMBEDDING_FIELD],
            'params': {
                'metric': 'cosine',
                'dimension': EMBEDDING_DIMENSION,
                'nLists': 2
            }
        })
        print(f'Created vector index: {VECTOR_INDEX_NAME}')
    except Exception as e:
        print(f'Error creating vector index: {e}')
    
    # Try a vector search
    try:
        # Generate query embedding
        query_embedding = generate_embedding()
        
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
            bind_vars={'query_embedding': query_embedding}
        )
        
        results = [doc for doc in cursor]
        print(f'Vector search found {len(results)} documents')
        
        if results:
            print('✅ Vector search is working!')
        else:
            print('⚠️ Vector search returned no results')
            
    except Exception as e:
        print(f'❌ Vector search failed: {e}')

if __name__ == '__main__':
    main()
