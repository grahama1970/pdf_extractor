#!/usr/bin/env python3

from pdf_extractor.arangodb_borked.connection import get_db

def main():
    db = get_db()
    print('Connected to ArangoDB')
    
    collection = db.collection('pdf_documents')
    print('Connected to collection')
    
    indexes = collection.indexes()
    print(f'Found {len(indexes)} indexes:')
    for idx in indexes:
        print(f' - {idx.get("name")}: {idx.get("type")}')
    
    # Try to create vector index
    try:
        collection.add_index({
            'type': 'vector',
            'name': 'vector_test',
            'fields': ['embeddings'],
            'params': {
                'metric': 'cosine',
                'dimension': 384
            }
        })
        print('Created vector index')
    except Exception as e:
        print(f'Error creating vector index: {e}')
    
    print('Done')

if __name__ == '__main__':
    main()
