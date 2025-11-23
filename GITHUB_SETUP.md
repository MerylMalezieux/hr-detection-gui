# Setting Up GitHub Repository

This guide will help you push the Heart Rate Detection GUI project to GitHub.

## Prerequisites

1. **Git installed** - If not installed, download from: https://git-scm.com/download/win
2. **GitHub account** - Create one at: https://github.com

## Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it (e.g., `hr-detection-gui` or `heart-rate-detection`)
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Initialize Git in Your Project

### Option A: Using Anaconda Prompt (Recommended for Network Drives)

1. Open **Anaconda Prompt**
2. Map the network drive:
   ```cmd
   net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
   ```
3. Navigate to the project:
   ```cmd
   Z:
   cd "Z:\"
   ```

4. Initialize git repository:
   ```cmd
   git init
   ```

5. Add all files:
   ```cmd
   git add .
   ```

6. Make your first commit:
   ```cmd
   git commit -m "Initial commit: Heart Rate Detection GUI"
   ```

7. Add the remote repository (replace YOUR_USERNAME and REPO_NAME):
   ```cmd
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   ```

8. Push to GitHub:
   ```cmd
   git branch -M main
   git push -u origin main
   ```

### Option B: Using Git Bash

1. Open **Git Bash**
2. Navigate to your project directory
3. Follow steps 4-8 from Option A

## Step 3: Authentication

When you push, GitHub will ask for authentication. You have two options:

### Option 1: Personal Access Token (Recommended)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name and select scopes: `repo` (full control)
4. Copy the token
5. When Git asks for password, paste the token instead

### Option 2: GitHub CLI

Install GitHub CLI and authenticate:
```cmd
gh auth login
```

## Step 4: Verify

1. Go to your GitHub repository page
2. You should see all your files there
3. The README.md should be visible

## Future Updates

After making changes to your code:

```cmd
# Navigate to project (if not already there)
Z:
cd "Z:\"

# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Common Issues

### Issue: "fatal: not a git repository"
**Solution**: Make sure you're in the project directory and run `git init`

### Issue: "fatal: remote origin already exists"
**Solution**: Remove and re-add:
```cmd
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### Issue: Authentication failed
**Solution**: Use a Personal Access Token instead of password

### Issue: UNC path not supported
**Solution**: Always map the network drive first using `net use Z: ...`

## Recommended Repository Structure

Your repository should include:
- ✅ `hr_detection_gui/` - Main package
- ✅ `main.py` - Entry point
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `setup_conda.bat` / `setup_env.ps1` - Setup scripts
- ❌ `venv/` - Excluded (in .gitignore)
- ❌ `*.npy` - Excluded (in .gitignore)
- ❌ `*.abf` - Excluded (in .gitignore)

## Adding a License

If you want to add a license:

1. Go to your repository on GitHub
2. Click "Add file" → "Create new file"
3. Name it `LICENSE`
4. GitHub will suggest templates - choose one (MIT, Apache, etc.)
5. Commit the file

## Collaborating

To allow others to contribute:

1. Go to repository Settings → Collaborators
2. Add people by username or email
3. They can clone with: `git clone https://github.com/YOUR_USERNAME/REPO_NAME.git`

