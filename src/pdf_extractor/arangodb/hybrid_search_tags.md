# Implementing Tag Filtering in Hybrid Search

This documentation describes how tag filtering was implemented in the hybrid search functionality,
allowing the agent to narrow search results by specifying one or more tags.

## Implementation Overview

1. **CLI Interface**: The CLI already had a `--tags` parameter in `cli_search_hybrid`, but its functionality was not fully implemented.

2. **Integration Changes**:
   - Added tag filter support to the `hybrid_search` function in `hybrid.py`
   - Updated `_fetch_bm25_candidates` in `bm25.py` to handle tag filtering
   - Updated `_fetch_semantic_candidates` in `semantic.py` to handle tag filtering
   - Created a demonstration in `tag_filter_demo.py`

## Usage for Agents

Agents can use the tag filtering feature when hybrid search returns too many unrelated results:

```bash
# Without tag filtering - may return diverse results
python -m src.pdf_extractor.arangodb.cli search hybrid "database optimization"

# With tag filtering - narrows to specific domain
python -m src.pdf_extractor.arangodb.cli search hybrid "database optimization" --tags "postgres,performance"
```

## Agent Decision Process

1. When initial search results are too broad, agent evaluates:
   - Are there common tags among the most relevant results?
   - Would focusing on specific tags improve relevance?

2. Decision criteria:
   - Use tag filtering when results span multiple topics/domains
   - Filter by the most relevant tags for the query context
   - Consider filtering by multiple tags for higher precision

3. After filtering, assess if relationship creation is beneficial between the filtered results

## Benefits for Agents

- **Improved Precision**: Tag filtering allows agents to find more focused results
- **Domain Specificity**: Can narrow searches to specific technologies or concepts
- **Better Relationship Suggestions**: More relevant document pairs for relationship creation
- **Workflow Efficiency**: Multi-step refinement process (broad → filtered → relationships)

## Example Workflow

1. Agent receives query: "optimize database performance"
2. Initial search returns documents about various databases
3. Agent notices multiple database types (MySQL, PostgreSQL, etc.)
4. Agent decides to filter: `--tags postgresql`
5. Filter produces focused results about PostgreSQL optimization
6. Agent identifies potential relationship between basic and advanced optimization techniques
7. Agent creates PREREQUISITE relationship between these documents
