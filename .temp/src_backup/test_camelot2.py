import os
import sys
import camelot

# Test file path
pdf_path = 'src/mcp_doc_retriever/context7/pdf_extractor/input/BHT_CV32A65X.pdf'

print(f'Testing extraction from {pdf_path}')
print(f'File exists: {os.path.exists(pdf_path)}')

# Try both extraction methods on just pages 1-2
for flavor in ['lattice', 'stream']:
    print(f'\nTrying {flavor} extraction on pages 1-2:')
    try:
        tables = camelot.read_pdf(pdf_path, pages='1-2', flavor=flavor)
        print(f'Found {len(tables)} tables')
        for i, table in enumerate(tables):
            print(f'  Table {i+1}: page={table.page}, rows={len(table.data)}, cols={len(table.data[0]) if table.data else 0}')
            print(f'  Accuracy: {table.parsing_report.get(accuracy, 0)}%')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
