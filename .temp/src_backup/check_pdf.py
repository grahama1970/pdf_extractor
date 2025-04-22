import os
import sys
from pypdf import PdfReader

# Test file path
pdf_path = 'src/mcp_doc_retriever/context7/pdf_extractor/input/BHT_CV32A65X.pdf'

print(f'Testing PDF: {pdf_path}')
print(f'File exists: {os.path.exists(pdf_path)}')
print(f'File size: {os.path.getsize(pdf_path)} bytes')

try:
    reader = PdfReader(pdf_path)
    print(f'Number of pages: {len(reader.pages)}')
    if len(reader.pages) > 0:
        print(f'Page 1 size: {reader.pages[0].mediabox}')
except Exception as e:
    print(f'Error opening PDF: {e}')
    import traceback
    traceback.print_exc()
