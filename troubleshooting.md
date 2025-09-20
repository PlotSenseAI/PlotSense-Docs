# Troubleshooting Guide

This guide helps you resolve common issues when using PlotSense.

## Installation Issues

### Problem: `pip install plotsense` fails

**Common Causes and Solutions:**

1. **Permission denied error**
   ```bash
   # Solution: Install for current user only
   pip install --user plotsense

   # Or use virtual environment (recommended)
   python -m venv plotsense_env
   source plotsense_env/bin/activate  # On Windows: plotsense_env\Scripts\activate
   pip install plotsense
   ```

2. **Network/proxy issues**
   ```bash
   # Solution: Use trusted hosts
   pip install --trusted-host pypi.org --trusted-host pypi.python.org plotsense

   # For corporate proxies
   pip install --proxy http://user:password@proxy.server:port plotsense
   ```

3. **Python version compatibility**
   ```bash
   # Check Python version
   python --version

   # PlotSense requires Python 3.7+
   # Upgrade Python if needed
   ```

4. **Dependency conflicts**
   ```bash
   # Create fresh environment
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install plotsense
   ```

### Problem: Import error after installation

```python
# Error: ModuleNotFoundError: No module named 'plotsense'

# Solutions:
# 1. Check if you're in the right environment
import sys
print(sys.executable)

# 2. Reinstall PlotSense
pip uninstall plotsense
pip install plotsense

# 3. Check installation location
pip show plotsense
```

## API and Authentication Issues

### Problem: API key not found

```python
# Error: "GROQ_API_KEY environment variable not found"

# Solutions:
# 1. Set environment variable
import os
os.environ["GROQ_API_KEY"] = "your-api-key-here"

# 2. Check if variable is set
print(os.getenv("GROQ_API_KEY"))

# 3. Use .env file
from dotenv import load_dotenv
load_dotenv()
```

### Problem: Invalid API key

```python
# Error: "Authentication failed" or "Invalid API key"

# Solutions:
# 1. Verify your API key at https://console.groq.com
# 2. Regenerate API key if needed
# 3. Check for extra spaces or characters
api_key = os.getenv("GROQ_API_KEY").strip()
```

### Problem: API rate limit exceeded

```python
# Error: "Rate limit exceeded" or 429 status code

# Solutions:
# 1. Add delays between requests
import time
suggestions = ps.recommender(df)
time.sleep(1)  # Wait 1 second
plot = ps.plotgen(df, suggestions.iloc[0])

# 2. Configure retry settings
ps.configure(
    max_retries=3,
    retry_delay=2.0,
    rate_limit_delay=0.5
)

# 3. Upgrade your Groq plan for higher limits
```

### Problem: Network connectivity issues

```python
# Error: "Connection timeout" or "Network error"

# Solutions:
# 1. Check internet connection
import requests
try:
    response = requests.get("https://api.groq.com", timeout=10)
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")

# 2. Configure longer timeout
ps.configure(api_timeout=60)

# 3. Check firewall/proxy settings
```

## Data-Related Issues

### Problem: No recommendations generated

```python
# Error: Empty recommendations DataFrame

# Solutions:
# 1. Check data requirements
print(f"Data shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values: {df.isnull().sum()}")

# Minimum requirements:
# - At least 2 rows
# - At least 1 meaningful column
# - Some non-null values

# 2. Clean your data
df_clean = df.dropna()
if len(df_clean) < 2:
    print("Insufficient data after cleaning")

# 3. Check column types
# Ensure you have numeric or categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"Numeric columns: {list(numeric_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")
```

### Problem: Poor quality recommendations

```python
# Solutions:
# 1. Improve data quality
# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values appropriately
df = df.fillna(df.mean())  # For numeric columns
df = df.fillna(df.mode().iloc[0])  # For categorical columns

# 2. Use meaningful column names
df.columns = ['meaningful_name_1', 'meaningful_name_2', ...]

# 3. Ensure appropriate data types
df['category_column'] = df['category_column'].astype('category')
df['date_column'] = pd.to_datetime(df['date_column'])

# 4. Reduce categorical cardinality
# Limit categories to reasonable number (< 20)
top_categories = df['category_col'].value_counts().head(10).index
df['category_col'] = df['category_col'].where(
    df['category_col'].isin(top_categories), 'Other'
)
```

### Problem: Large dataset performance issues

```python
# Error: Slow performance or memory issues

# Solutions:
# 1. Sample large datasets
if len(df) > 10000:
    df_sample = df.sample(n=5000, random_state=42)
    suggestions = ps.recommender(df_sample)

# 2. Configure performance settings
ps.configure(
    max_data_points=5000,
    sampling_strategy='random',
    sample_size=3000
)

# 3. Optimize data types
# Use categorical for string columns with few unique values
df['category'] = df['category'].astype('category')

# Use appropriate numeric types
df['integer_col'] = df['integer_col'].astype('int32')
df['float_col'] = df['float_col'].astype('float32')
```

## Visualization Issues

### Problem: Plot generation fails

```python
# Error: "Plot generation failed" or matplotlib errors

# Solutions:
# 1. Check data compatibility
suggestion = suggestions.iloc[0]
required_cols = [suggestion.get('x_column'), suggestion.get('y_column')]
missing_cols = [col for col in required_cols if col and col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")

# 2. Handle invalid data
# Remove infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 3. Check column data types
x_col = suggestion.get('x_column')
if x_col and not pd.api.types.is_numeric_dtype(df[x_col]):
    if suggestion['plot_type'] in ['scatter', 'line']:
        print(f"Warning: {x_col} is not numeric for {suggestion['plot_type']} plot")
```

### Problem: Plots don't display

```python
# Solutions:
# 1. Enable matplotlib backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'

# 2. For Jupyter notebooks
%matplotlib inline
# or
%matplotlib widget

# 3. Explicitly show plots
plot = ps.plotgen(df, suggestion)
plot.show()

# 4. Save plot instead
plot.savefig('my_plot.png', dpi=300, bbox_inches='tight')
```

### Problem: Poor plot quality or formatting

```python
# Solutions:
# 1. Customize plot parameters
plot = ps.plotgen(
    df,
    suggestion,
    figsize=(12, 8),
    title="Custom Title",
    style='seaborn-v0_8'
)

# 2. Configure global defaults
ps.configure(
    default_figsize=(10, 6),
    default_dpi=150,
    default_style='seaborn-v0_8'
)

# 3. Post-process the plot
import matplotlib.pyplot as plt
plot = ps.plotgen(df, suggestion)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()
```

## Explanation Issues

### Problem: Poor or generic explanations

```python
# Solutions:
# 1. Use custom prompts
explanation = ps.explainer(
    plot,
    custom_prompt="Focus on business insights and actionable recommendations"
)

# 2. Use multiple iterations
explanation = ps.explainer(
    plot,
    custom_prompt="Provide detailed statistical analysis",
    iterations=2
)

# 3. Provide domain context
domain_prompt = """
Analyze this healthcare data visualization focusing on:
- Patient outcomes and trends
- Statistical significance of patterns
- Clinical implications
- Recommendations for healthcare providers
"""
explanation = ps.explainer(plot, custom_prompt=domain_prompt)
```

### Problem: Explanation generation fails

```python
# Error: "Failed to generate explanation"

# Solutions:
# 1. Check API connectivity
import requests
try:
    response = requests.get("https://api.groq.com/openai/v1/models",
                           headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"})
    print(f"API Status: {response.status_code}")
except Exception as e:
    print(f"API Error: {e}")

# 2. Simplify the plot
# Ensure plot is not too complex
simple_suggestion = {
    'plot_type': 'scatter',
    'x_column': 'simple_x',
    'y_column': 'simple_y'
}
simple_plot = ps.plotgen(df[['simple_x', 'simple_y']], simple_suggestion)

# 3. Check plot object
if plot is None:
    print("Plot object is None - regenerate plot first")
```

## Performance Issues

### Problem: Slow response times

```python
# Solutions:
# 1. Enable caching
ps.configure(
    cache_recommendations=True,
    cache_directory='~/.plotsense_cache'
)

# 2. Reduce data size
df_sample = df.sample(n=min(1000, len(df)))

# 3. Configure shorter timeouts
ps.configure(
    api_timeout=15,
    max_retries=1
)

# 4. Use async processing (if available)
import asyncio

async def process_multiple_datasets(datasets):
    tasks = []
    for df in datasets:
        tasks.append(ps.recommender_async(df))
    return await asyncio.gather(*tasks)
```

### Problem: Memory usage issues

```python
# Solutions:
# 1. Process data in chunks
def process_large_dataset(df, chunk_size=1000):
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        suggestions = ps.recommender(chunk)
        results.append(suggestions)
    return pd.concat(results, ignore_index=True)

# 2. Clean up plot objects
plot = ps.plotgen(df, suggestion)
# Use the plot
plot.show()
# Clean up
plt.close(plot)
del plot

# 3. Configure memory limits
ps.configure(
    memory_limit_mb=256,
    cleanup_plots=True
)
```

## Environment-Specific Issues

### Problem: Issues in Jupyter Notebooks

```python
# Solutions:
# 1. Restart kernel and reimport
# Kernel -> Restart & Clear Output

# 2. Configure for Jupyter
%matplotlib inline
import plotsense as ps
ps.configure(
    show_plots=True,
    default_figsize=(10, 6),
    verbose=False
)

# 3. Handle widget issues
# For interactive plots
%matplotlib widget
```

### Problem: Issues in Docker containers

```dockerfile
# Dockerfile solutions
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Set matplotlib backend
ENV MPLBACKEND=Agg

# Install PlotSense
RUN pip install plotsense

# Set API key
ENV GROQ_API_KEY=your-api-key-here
```

### Problem: Issues in cloud environments

```python
# Solutions for cloud platforms
# 1. AWS Lambda
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

# 2. Google Colab
# Install and configure
!pip install plotsense
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

# 3. Azure Functions
# Use serverless-friendly configuration
ps.configure(
    cache_recommendations=False,
    auto_save_plots=True,
    show_plots=False
)
```

## Debugging and Diagnostics

### Enable Debug Mode

```python
import plotsense as ps
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
ps.configure(
    verbose=True,
    log_level='DEBUG'
)

# Run with debug info
try:
    suggestions = ps.recommender(df)
    print(f"Generated {len(suggestions)} suggestions")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### System Information

```python
import plotsense as ps
import sys
import pandas as pd
import matplotlib
import platform

# Print system info
print("=== System Information ===")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PlotSense version: {ps.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Check API key
import os
api_key = os.getenv('GROQ_API_KEY')
print(f"API key configured: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API key length: {len(api_key)} characters")
```

### Test Installation

```python
import plotsense as ps
import pandas as pd
import numpy as np

def test_plotsense_installation():
    """Test PlotSense installation and basic functionality"""
    try:
        # Test data creation
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50),
            'category': np.random.choice(['A', 'B', 'C'], 50)
        })
        print("✓ Test data created successfully")

        # Test recommendations
        suggestions = ps.recommender(df)
        print(f"✓ Generated {len(suggestions)} recommendations")

        # Test plot generation
        if len(suggestions) > 0:
            plot = ps.plotgen(df, suggestions.iloc[0])
            print("✓ Plot generated successfully")

            # Test explanation
            explanation = ps.explainer(plot)
            print("✓ Explanation generated successfully")
            print(f"Sample explanation: {explanation[:100]}...")

        print("\n✅ All tests passed! PlotSense is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Run test
test_plotsense_installation()
```

## Getting Additional Help

### Before Seeking Help

1. **Check this troubleshooting guide** for similar issues
2. **Review error messages** carefully for specific details
3. **Test with simple data** to isolate the problem
4. **Check your internet connection** for API-related issues

### Where to Get Help

1. **GitHub Issues**: [PlotSense GitHub Repository](https://github.com/PlotSenseAI/PlotSense/issues)
   - Search existing issues first
   - Provide detailed error messages and code samples
   - Include system information (Python version, OS, etc.)

2. **Documentation**: Review other documentation pages
   - [Getting Started](getting-started.md)
   - [API Reference](api-reference.md)
   - [Configuration](configuration.md)

3. **Community**:
   - Stack Overflow (tag: plotsense)
   - Discord/Slack communities (if available)

### Reporting Bugs

When reporting bugs, include:

```python
# Bug report template
"""
**Environment Information:**
- Python version:
- PlotSense version:
- Operating System:
- Installation method: pip/conda/source

**Issue Description:**
Brief description of the problem

**Reproduction Steps:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Error Message:**
```
Full error message and traceback
```

**Sample Code:**
```python
# Minimal code that reproduces the issue
import plotsense as ps
# ... rest of code
```

**Additional Context:**
Any other relevant information
"""
```

Remember: The more information you provide, the easier it is to help resolve your issue!