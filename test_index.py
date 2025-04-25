#!/usr/bin/env python3

from pdf_extractor.arangodb_borked.connection import get_db

# Constants
PDF_COLLECTION_NAME = 'pdf_documents'
EMBEDDING_FIELD = 'embeddings'
EMBEDDING_DIMENSION = 384
VECTOR_INDEX_NAME = 'vector_index_fixed'

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
    
    print('Done')

if __name__ == '__main__':
    main()
