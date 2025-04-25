#!/usr/bin/env python3

from pdf_extractor.arangodb_borked.connection import get_db
import os
import re

def update_semantic_search():
    # File path
    file_path = '/home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/search_functions.py'
    print(f'Updating semantic search in {file_path}')
    
    # New implementation
    new_semantic_search = '''
def semantic_search(db, query_embedding, limit=10):
    