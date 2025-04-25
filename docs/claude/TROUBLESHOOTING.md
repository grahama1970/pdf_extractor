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