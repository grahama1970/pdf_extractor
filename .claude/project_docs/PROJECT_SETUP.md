# Project Setup Guide: pdf_extractor

This guide outlines the steps to set up the `pdf_extractor` project locally and push it to GitHub.

## Prerequisites

- Linux/Ubuntu system
- Git installed
- GitHub account

## Installation Steps

### 1. Install GitHub CLI (gh)

The GitHub CLI allows you to interact with GitHub from your terminal:

```bash
# Add GitHub CLI's GPG key
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

# Add GitHub CLI repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# Update package list and install gh
sudo apt update
sudo apt install gh -y

# Verify installation
gh --version
```

### 2. Authenticate with GitHub

```bash
# Start the login process
gh auth login

# Follow the prompts:
# - Choose SSH for Git operations (if you have SSH keys set up)
# - Choose your preferred authentication method (browser or token)
```

### 3. Initialize Local Repository

Navigate to your project directory and initialize Git:

```bash
# Navigate to project directory
cd /home/graham/workspace/experiments/pdf_extractor

# Initialize Git repository (if not already done)
git init

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit"
```

### 4. Create and Push to GitHub Repository

Use GitHub CLI to create a new repository and push your code:

```bash
# Create repository and push in one command
gh repo create pdf_extractor --source=. --public --push
```

This command:
- Creates a new public repository named `pdf_extractor`
- Uses the current directory (`.`) as the source
- Automatically pushes your local code to the new repository

## Command Parameters Explained

- `pdf_extractor`: The name of your repository on GitHub
- `--source=.`: Specifies that the source code is in the current directory
- `--public`: Makes the repository publicly visible (use `--private` for private repos)
- `--push`: Automatically sets up the remote and pushes your code

## Additional Options

- To add a description: `--description="Your description here"`
- To make it private: Replace `--public` with `--private`
- To make it internal (for organizations): Replace `--public` with `--internal`

## Verification

After pushing, you can verify your repository is set up correctly:

```bash
# Check remote repository status
gh repo view

# View repository in browser
gh repo view --web
```

## Troubleshooting

1. If `gh auth login` fails:
   - Ensure you have a stable internet connection
   - Try authenticating with a personal access token instead of browser

2. If push fails:
   - Verify you have the correct permissions
   - Check if you need to configure SSH keys
   - Ensure you have made at least one commit

3. For SSH issues:
   - Verify SSH keys are properly set up: `ssh -T git@github.com`
   - Add SSH key if needed: `ssh-keygen -t ed25519 -C "your_email@example.com"`