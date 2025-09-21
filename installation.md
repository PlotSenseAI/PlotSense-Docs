# Installation Guide

This guide covers all the ways to install PlotSense and set up your environment.

## Quick Installation

The fastest way to get started with PlotSense is using pip:

```bash
pip install plotsense
```

## System Requirements

### Python Version
- **Python 3.7 or higher** is required
- Python 3.8+ is recommended for best performance

### Operating Systems
PlotSense is compatible with:
- Windows 10+
- macOS 10.14+
- Linux (Ubuntu 18.04+, CentOS 7+, and other modern distributions)

## Installation Methods

### 1. Install via pip (Recommended)

```bash
# Install the latest stable version
pip install plotsense

# Install a specific version
pip install plotsense==1.0.0

# Upgrade to the latest version
pip install --upgrade plotsense
```

### 2. Install via conda

```bash
# Install from conda-forge (if available)
conda install -c conda-forge plotsense

# Or using mamba for faster installation
mamba install -c conda-forge plotsense
```

### 3. Install from Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/PlotSenseAI/PlotSense.git
cd PlotSense

# Install in development mode
pip install -e .
```

## Dependencies

PlotSense automatically installs the following dependencies:

### Required Dependencies
- **pandas** (>= 1.0.0) - Data manipulation and analysis
- **matplotlib** (>= 3.0.0) - Plotting library
- **seaborn** (>= 0.11.0) - Statistical data visualization
- **requests** (>= 2.25.0) - HTTP library for API calls
- **groq** - Groq API client

### Optional Dependencies
- **numpy** (>= 1.18.0) - Numerical computing (usually installed with pandas)
- **jupyter** - For Jupyter notebook support

## API Key Setup

PlotSense requires a Groq API key for AI-powered features. Here's how to set it up:

### 1. Get Your API Key

1. Visit [Groq Console](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy your API key

### 2. Configure Your API Key

Choose one of the following methods:

#### Method 1: Environment Variable (Recommended)

**On Windows:**
```cmd
set GROQ_API_KEY=your-api-key-here
```

**On macOS/Linux:**
```bash
export GROQ_API_KEY="your-api-key-here"
```

**To make it permanent, add to your shell profile:**
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.bash_profile
echo 'export GROQ_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### Method 2: Python Configuration

```python
import os
os.environ["GROQ_API_KEY"] = "your-api-key-here"
```

#### Method 3: Configuration File

Create a `.env` file in your project directory:
```
GROQ_API_KEY=your-api-key-here
```

Then load it in your Python script:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Verification

Verify your installation by running:

```python
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer

# Test with sample data
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
suggestions = recommender(df)
print(f"Found {len(suggestions)} suggestions")
```

## Virtual Environment Setup

We recommend using a virtual environment:

### Using venv

```bash
# Create virtual environment
python -m venv plotsense-env

# Activate (Windows)
plotsense-env\Scripts\activate

# Activate (macOS/Linux)
source plotsense-env/bin/activate

# Install PlotSense
pip install plotsense
```

### Using conda

```bash
# Create conda environment
conda create -n plotsense python=3.9

# Activate environment
conda activate plotsense

# Install PlotSense
pip install plotsense
```

## Docker Installation

For containerized environments:

```dockerfile
FROM python:3.9-slim

# Install PlotSense
RUN pip install plotsense

# Set API key (replace with your key)
ENV GROQ_API_KEY="your-api-key-here"

# Your application code
COPY . /app
WORKDIR /app

CMD ["python", "your_script.py"]
```

## Troubleshooting Installation

### Common Issues

1. **Permission Denied Error**
   ```bash
   pip install --user plotsense
   ```

2. **Network/Proxy Issues**
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org plotsense
   ```

3. **Dependency Conflicts**
   ```bash
   # Create a fresh environment
   python -m venv fresh-env
   source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows
   pip install plotsense
   ```

4. **Missing API Key Error**
   - Ensure your GROQ_API_KEY is properly set
   - Check for typos in your API key
   - Verify your API key is active in Groq Console

### Getting Help

If you encounter installation issues:

1. Check our [Troubleshooting Guide](troubleshooting.md)
2. Search existing issues on [GitHub](https://github.com/PlotSenseAI/PlotSense/issues)
3. Create a new issue with:
   - Your Python version (`python --version`)
   - Your OS and version
   - The complete error message
   - Installation method used

## Next Steps

Once installed, check out:
- [Getting Started Guide](getting-started.md) - Your first PlotSense visualization
- [Examples](examples.md) - Comprehensive usage examples
- [API Reference](api-reference.md) - Detailed function documentation
