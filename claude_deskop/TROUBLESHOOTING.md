# Troubleshooting Guide

This document contains solutions to common issues you might encounter while working on this project.

## SSH File Creation Issues

### Problem: Special Characters in Python Files

When creating Python files via SSH that contain special characters (string literals, variables, etc.), you might encounter syntax errors or unexpected behavior because the shell interprets special characters before passing them to the remote file.

### Solution: Use Single-Quoted Heredoc

**Always use single quotes around the heredoc delimiter (EOF)** to prevent shell expansion:

```bash
ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 'cat > /path/to/file.py' <<'EOF'
#!/usr/bin/env python3

# Python code with any special characters
print("Double quotes work fine")
print('Single quotes work fine')
multiline = """
Triple-quoted strings with "double quotes" and 'single quotes'
and $variables and \backslashes all work correctly
"""

## Verifying ArangoDB Connection

### Problem: Unsure if ArangoDB is Working Correctly

If you're experiencing issues with database operations, it's helpful to verify that your ArangoDB connection is working correctly.

### Solution: Use these Test Commands

To verify the basic ArangoDB connection, run:

```bash
python -c "from arango import ArangoClient; client = ArangoClient(hosts='http://192.168.86.49:8529/'); db = client.db('_system', username='root', password='openSesame'); print(f'Connected to ArangoDB {db.version()}')"
```

To check the pdf_extractor database and its collections:

```bash
python -c "from arango import ArangoClient; client = ArangoClient(hosts='http://192.168.86.49:8529/'); db = client.db('pdf_extractor', username='root', password='openSesame'); collections = [c[\"name\"] for c in db.collections() if not c[\"name\"].startswith(\"_\")]; print('Collections:', collections)"
```

These commands should be run from within your virtual environment after installing all required dependencies.

### Expected Output

The first command should output something like:
```
Connected to ArangoDB 3.12.4
```

The second command should list the available collections:
```
Collections: ['relationships', 'documents', 'lessons_learned', 'pdf_documents', ...]
```

If you get authentication errors, verify your username and password. If the database doesn't exist, you may need to create it first.
