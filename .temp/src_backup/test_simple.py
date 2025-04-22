# Simple test for tables_merger.py
from tables_merger import calculate_table_similarity, merge_multi_page_tables

print('Testing table merging implementation')

# Create test tables
test_tables = [
    {
        'page': 1,
        'data': [
            ['Col1', 'Col2', 'Col3'],
            ['data1', 'data2', 'data3']
        ],
        'bbox': (50, 700, 550, 750),
        'rows': 2,
        'cols': 3
    },
    {
        'page': 2,
        'data': [
            ['Col1', 'Col2', 'Col3'],
            ['data4', 'data5', 'data6'],
            ['data7', 'data8', 'data9']
        ],
        'bbox': (50, 700, 550, 750),
        'rows': 3,
        'cols': 3
    }
]

# Calculate similarity
similarity = calculate_table_similarity(test_tables[0], test_tables[1])
print('Similarity:', similarity)

# Merge tables
merged = merge_multi_page_tables(test_tables)
print('Merged tables count:', len(merged))

if len(merged) == 1:
    print('Tables merged successfully')
    print('Rows in merged table:', merged[0]['rows'])
    print('Is multi-page:', merged[0].get('is_multi_page'))
    print('Page range:', merged[0].get('page_range'))
else:
    print('Tables were not merged as expected')

print('Test completed')
