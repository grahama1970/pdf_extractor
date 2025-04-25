#!/bin/bash
python -c "import numpy, pandas, matplotlib, scipy, sklearn" &
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug