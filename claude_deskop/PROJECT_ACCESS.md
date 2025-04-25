You are the user 'graham' logging into a Ubuntu enviroment
With your tools execute the command:
{
  `command`: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 \"cd /home/graham/workspace/experiments/pdf_extractor/ && source .venv/bin/activate && python -m src.pdf_extractor.arangodb.search_cli --help\"`
}

When creating files via SSH that contain special characters (especially Python code with string literals), use the following command pattern with single-quoted heredoc delimiter to prevent shell expansion:

ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 'cat > /path/to/file.py' <<'EOF'
# Python code here with any special characters
# No need to escape $, ", ', or \
EOF

With your tools execute the command:
{
  `command`: `ssh -i ~/.ssh/id_ed25519_wsl2 graham@192.168.86.49 "cd /home/graham/workspace/experiments/pdf_extractor/ && source .venv/bin/activate && python -m src.pdf_extractor.arangodb.search_cli --help"`
}