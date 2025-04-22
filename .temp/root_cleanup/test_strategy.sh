#!/bin/bash
cd /home/graham/workspace/experiments/pdf_extractor
source .venv/bin/activate

echo TESTING DIFFERENT MERGE STRATEGIES
echo =================================

# Test with conservative strategy
echo -e nStrategy: conservative
python -c 