# Let's rewrite the problematic line
    print(f"Found {len(results.get('results', []))} results")
    
    for i, result in enumerate(results.get('results', []), 1):
        print(f"{i}. [{result.get('collection', 'unknown')}] Score: {result.get('score', 0):.4f}")
        print(f"   ID: {result.get('_id', 'unknown')}")
        content = result.get('content', '')
        if len(content) > 100:
            print(f"   Content: {content[:100]}...")
        else:
            print(f"   Content: {content}")
