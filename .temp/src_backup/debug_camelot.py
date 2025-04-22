#!/usr/bin/env python3
# Debug script to check camelot installation

import sys
import traceback

print('Python version:', sys.version)
print('Python executable:', sys.executable)

print('\nAttempting to import camelot:')
try:
    import camelot
    print('Successfully imported camelot module')
    try:
        print('Camelot version:', camelot.__version__)
    except AttributeError:
        print('Could not determine camelot version')
except ImportError as e:
    print('Failed to import camelot:', e)
    traceback.print_exc()

print('\nChecking camelot dependencies:')
dependencies = ['pandas', 'numpy', 'cv2', 'pdfminer', 'ghostscript', 'openpyxl']

for dep in dependencies:
    try:
        module = __import__(dep)
        print(dep, 'is installed')
    except ImportError:
        print(dep, 'is NOT installed')
