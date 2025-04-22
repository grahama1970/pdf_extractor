import os
import sys
import camelot
from pathlib import Path

# Test file
pdf_path = Path('src/mcp_doc_retriever/context7/pdf_extractor/input/BHT_CV32A65X.pdf')

print(f'Testing PDF: {pdf_path}')
print(f'File exists: {os.path.exists(pdf_path)}')

# Try to extract tables
try:
    tables = camelot.read_pdf(str(pdf_path), pages='1-2')
    print(f'Extracted {len(tables)} tables')
    
    # Process tables into dict form
    processed_tables = []
    for table in tables:
        processed = {
            'page': int(table.page),
            'data': table.data,
            'rows': len(table.data),
            'cols': len(table.data[0]) if table.data else 0,
            'accuracy': table.parsing_report.get('accuracy', 0)
        }
        processed_tables.append(processed)
        
    # Check for multi-page tables
    if len(processed_tables) >= 2:
        t1 = processed_tables[0]
        t2 = processed_tables[1]
        
        # Simple check for matching headers
        if t1['data'] and t2['data'] and len(t1['data'][0]) == len(t2['data'][0]):
            headers1 = t1['data'][0]
            headers2 = t2['data'][0]
            
            matching_count = sum(1 for h1, h2 in zip(headers1, headers2) if h1.strip() == h2.strip())
            similarity = matching_count / len(headers1) if headers1 else 0
            
            print(f'Header similarity between first two tables: {similarity:.2f}')
            
            if similarity >= 0.7:
                print('These could be merged as a multi-page table')
                
                # Merge the data
                merged_data = t1['data'] + t2['data'][1:]  # Skip second header
                print(f'Merged table would have {len(merged_data)} rows, {len(merged_data[0]) if merged_data else 0} columns')
    
except Exception as e:
    print(f'Error: {e}')
