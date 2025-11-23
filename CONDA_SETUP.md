# Conda Environment Setup Instructions

This guide will help you set up the Heart Rate Detection GUI project using Anaconda/Miniconda.

## Quick Setup

### Option 1: Using the Setup Script

1. Open **Anaconda Prompt**
2. Navigate to the project directory:
   ```cmd
   cd "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
   ```
   Note: If CMD doesn't support UNC paths, map the drive first:
   ```cmd
   net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
   Z:
   cd "Z:\"
   ```

3. Run the setup script:
   ```cmd
   setup_conda.bat
   ```

### Option 2: Manual Setup

1. Open **Anaconda Prompt**

2. Navigate to the project directory (or map the drive if needed):
   ```cmd
   net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
   Z:
   cd "Z:\"
   ```

3. Create a new conda environment:
   ```cmd
   conda create -n hr_detection_gui python=3.9 -y
   ```
   (You can change `python=3.9` to your preferred Python version, e.g., `python=3.10` or `python=3.11`)

4. Activate the environment:
   ```cmd
   conda activate hr_detection_gui
   ```

5. Upgrade pip:
   ```cmd
   python -m pip install --upgrade pip
   ```

6. Install the required packages:
   ```cmd
   pip install -r requirements.txt
   ```

## Using the Environment

### Activating the Environment

Every time you want to work on this project, open **Anaconda Prompt** and run:

```cmd
conda activate hr_detection_gui
```

Then navigate to your project directory (or use the mapped drive).

### Running the Application

Once the environment is activated:

```cmd
python main.py
```

### Deactivating the Environment

When you're done working:

```cmd
conda deactivate
```

## Managing the Environment

### List all conda environments:
```cmd
conda env list
```

### Remove the environment (if needed):
```cmd
conda env remove -n hr_detection_gui
```

### Export environment (for sharing):
```cmd
conda env export > environment.yml
```

### Create environment from file:
```cmd
conda env create -f environment.yml
```

## Troubleshooting

### If conda command is not recognized:
- Make sure you're using **Anaconda Prompt** (not regular CMD or PowerShell)
- Or initialize conda in your current shell:
  ```cmd
  C:\Users\YourUsername\Anaconda3\Scripts\activate.bat
  ```

### If you get package conflicts:
Try installing packages one by one or use conda instead of pip for some packages:
```cmd
conda install numpy pandas scipy matplotlib -y
pip install neo openpyxl
```

### If UNC path doesn't work:
Map the network drive first:
```cmd
net use Z: "\\LTAds\DataStor\malezieux_meryl\Postdoc\Analysis\Python_scripts\HR_GUI_Project"
Z:
cd "Z:\"
```

## Notes

- The conda environment is separate from your base Anaconda environment
- You can have multiple conda environments for different projects
- The environment name `hr_detection_gui` can be changed to whatever you prefer

