# File Operations and Research Capabilities

## SSH and File Operations
- **SSH Connection**: You can connect to the remote server using: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49`
- **File Access**: You have full permission to read from and write to files within the project directory at `/home/graham/workspace/experiments/pdf_extractor/`
- **Writing Files**: You can write to files using several methods:
  ```bash
  # Direct echo to remote file via SSH
  echo "content" | ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 "cat > /home/graham/workspace/experiments/pdf_extractor/path/to/file.txt"
  
  # Multiple line content to remote file 
  ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 "cat > /home/graham/workspace/experiments/pdf_extractor/path/to/file.txt << 'EOF'
  Line 1
  Line 2
  Line 3
  EOF"
  
  # Creating directories before writing files
  ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 "mkdir -p /home/graham/workspace/experiments/pdf_extractor/path/to/dir && cat > /home/graham/workspace/experiments/pdf_extractor/path/to/dir/file.txt << 'EOF'
  content
  EOF"
  ```

## Web Research with Perplexity
- **When to Use Perplexity**:
  1. Whenever you encounter an unfamiliar technology, API, or concept
  2. When you are unsure of how to implement a feature or solve a problem
  3. When you need up-to-date information about libraries, frameworks, or best practices
  4. If you encounter errors or unexpected behaviors you cannot immediately explain
  5. When you need to verify your understanding before implementing a solution

- **How to Use Perplexity**:
  1. Formulate clear, specific search queries
  2. Generate multiple search terms for the same concept to ensure comprehensive results
  3. Use the `web_search` tool or a similar research tool available in your environment
  4. Analyze results critically and synthesize information from multiple sources

## Troubleshooting
- If you encounter permission issues with SSH or file operations, verify:
  1. SSH key path and permissions: `chmod 600 ~/.ssh/id_ed25519_wsl2`
  2. Target directory permissions 
  3. File ownership issues
- For file write failures, try alternative methods such as creating temporary files and transferring them

## Best Practices
- Always test file operations with small changes before making large modifications
- Create backups of critical files before modifying them
- Use absolute paths rather than relative paths for clarity
- When confused about any capability or encountering errors, immediately use Perplexity to research possible solutions rather than making assumptions
- Document any file changes made to maintain a clear audit trail
