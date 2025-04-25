#!/bin/bash
# Fix missing quotes in the ArangoDB integration file
sed -i 's/logger.info(Created fulltext index for text field)/logger.info("Created fulltext index for text field")/g' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
sed -i 's/logger.info(Created hash index for type field)/logger.info("Created hash index for type field")/g' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
sed -i 's/logger.info(Created hash index for file_path field)/logger.info("Created hash index for file_path field")/g' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
sed -i 's/logger.info(Created skiplist index for page field)/logger.info("Created skiplist index for page field")/g' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
sed -i 's/logger.info(Created vector index for embeddings field)/logger.info("Created vector index for embeddings field")/g' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py
