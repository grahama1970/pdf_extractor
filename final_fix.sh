#!/bin/bash
# Fix the string formatting error in the ArangoDB integration

sed -i 's/print(f\x27{key.replace(_, ).title()}: {status}\x27)/print(f\x27{key.replace("_", " ").title()}: {status}\x27)/' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py

sed -i 's/total_testable = sum(1 for v in results.values() if v is not \x27not_tested\x27)/total_testable = sum(1 for v in results.values() if v != \x27not_tested\x27)/' /home/graham/workspace/experiments/pdf_extractor/src/pdf_extractor/arangodb/pdf_integration.py

echo Fixed string formatting errors

# Test the implementation
cd /home/graham/workspace/experiments/pdf_extractor && source .venv/bin/activate && python -m pdf_extractor.arangodb.pdf_integration
