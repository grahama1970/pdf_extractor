#!/usr/bin/env python3
import logging
from arango.exceptions import DocumentInsertError, DocumentUpdateError, DocumentDeleteError

logger = logging.getLogger(__name__)

def insert_document(collection, document):
    