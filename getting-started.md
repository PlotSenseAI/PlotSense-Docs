# Getting Started with PlotSense

This guide will help you get up and running with PlotSense in just a few minutes.

## Prerequisites

Before you begin, make sure you have:
- Python 3.7 or higher
- pandas library installed
- A Groq API key (required for AI features)

## Step 1: Installation

Install PlotSense using pip:

```bash
pip install plotsense
```

## Step 2: Set Up Your API Key

PlotSense requires a Groq API key for its AI-powered features. You can obtain a free API key from [Groq's website](https://console.groq.com).

### Setting Up Your API Key

You can configure your API key in several ways:

1. **Environment Variable (Recommended)**:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```

2. **In Your Python Code**:
   ```python
   import os
   os.environ["GROQ_API_KEY"] = "your-api-key-here"
   ```

## Step 3: Your First PlotSense Visualization

Let's create your first AI-powered visualization:

```python
import plotsense as ps
import pandas as pd

# Create sample data
data = {
    'x': [1, 2, 3, 4, 5],
    'y': [2, 5, 3, 8, 7],
    'category': ['A', 'B', 'A', 'B', 'A']
}
df = pd.DataFrame(data)

# Step 1: Get AI recommendations
print("Getting AI recommendations...")
suggestions = ps.recommender(df)
print(f"Found {len(suggestions)} visualization suggestions")

# Step 2: Generate a plot from the first suggestion
print("Generating plot...")
plot = ps.plotgen(df, suggestions.iloc[0])

# Step 3: Get AI explanation
print("Getting AI explanation...")
explanation = ps.explainer(plot)
print(f"Explanation: {explanation}")
```

## Understanding the Core Workflow

PlotSense follows a simple three-step workflow:

### 1. **Recommender** (`ps.recommender()`)
- Analyzes your dataset structure
- Returns up to 10 visualization suggestions
- Each suggestion includes plot type and recommended variables

### 2. **Plot Generator** (`ps.plotgen()`)
- Takes your DataFrame and a recommendation
- Generates the actual visualization
- Returns a plot object you can display or save

### 3. **Explainer** (`ps.explainer()`)
- Analyzes your generated plot
- Provides natural language insights
- Helps you understand patterns in your data

## Working with Real Data

Here's how to use PlotSense with your own datasets:

```python
import plotsense as ps
import pandas as pd

# Load your data
df = pd.read_csv("your_dataset.csv")

# Get recommendations
suggestions = ps.recommender(df)

# Explore different suggestions
for i, suggestion in suggestions.iterrows():
    print(f"Suggestion {i+1}: {suggestion['plot_type']} - {suggestion['description']}")

# Generate your preferred plot
selected_suggestion = suggestions.iloc[0]  # or choose any index
plot = ps.plotgen(df, selected_suggestion)

# Get insights
explanation = ps.explainer(plot)
print(explanation)
```

## Next Steps

Now that you have PlotSense working, explore:

- [API Reference](api-reference.md) - Detailed documentation of all functions
- [Examples](examples.md) - More comprehensive examples and use cases
- [Configuration](configuration.md) - Advanced configuration options
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Tips for Better Results

1. **Clean Data**: Ensure your data is properly formatted and doesn't have missing values in key columns
2. **Meaningful Column Names**: Use descriptive column names to help the AI understand your data
3. **Appropriate Data Size**: PlotSense works best with datasets that have between 10-10,000 rows
4. **Mixed Data Types**: Include both numerical and categorical columns for more visualization options