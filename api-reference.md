# API Reference

This page provides detailed documentation for all PlotSense functions and their parameters.

## Core Functions

### `plotsense.recommender()`

Analyzes a pandas DataFrame and returns AI-powered visualization recommendations.

**Signature:**
```python
plotsense.recommender(dataframe)
```

**Parameters:**
- `dataframe` (pandas.DataFrame): The dataset to analyze for visualization recommendations

**Returns:**
- `pandas.DataFrame`: A DataFrame containing up to 10 visualization suggestions with the following columns:
  - `plot_type`: The recommended plot type (scatter, bar, histogram, etc.)
  - `x_column`: Recommended column for x-axis
  - `y_column`: Recommended column for y-axis (if applicable)
  - `color_column`: Recommended column for color encoding (if applicable)
  - `description`: Human-readable description of the recommendation
  - `confidence`: Confidence score for the recommendation (0-1)

**Example:**
```python
from plotsense import recommender
import pandas as pd

df = pd.read_csv("sales_data.csv")
suggestions = recommender(df)
print(suggestions.head())
```

**Supported Data Types:**
- Numerical columns (int, float)
- Categorical columns (string, category)
- DateTime columns
- Boolean columns

---

### `plotsense.plotgen()`

Generates a visualization based on a DataFrame and a recommendation.

**Signature:**
```python
plotsense.plotgen(dataframe, suggestion, **kwargs)
```

**Parameters:**
- `dataframe` (pandas.DataFrame): The dataset to visualize
- `suggestion` (pandas.Series or dict): A recommendation from `ps.recommender()` or custom plot specification
- `**kwargs`: Additional customization options

**Returns:**
- `matplotlib.figure.Figure`: The generated plot figure

**Example:**
```python
from plotsense import plotgen
import pandas as pd

df = pd.read_csv("data.csv")
suggestions = recommender(df)
plot = plotgen(df, suggestions.iloc[0])
plot.show()
```

**Custom Suggestion Format:**
```python
custom_suggestion = {
    'plot_type': 'scatter',
    'x_column': 'age',
    'y_column': 'salary',
    'color_column': 'department'
}
plot = plotgen(df, custom_suggestion)
```

**Supported Plot Types:**
- `scatter`: Scatter plot
- `bar`: Vertical bar chart
- `barh`: Horizontal bar chart
- `histogram`: Histogram
- `boxplot`: Box plot
- `violinplot`: Violin plot
- `pie`: Pie chart
- `hexbin`: Hexagonal binning plot

**Additional Parameters:**
- `figsize` (tuple): Figure size in inches, e.g., `(10, 6)`
- `title` (str): Custom title for the plot
- `xlabel` (str): Custom x-axis label
- `ylabel` (str): Custom y-axis label
- `color_palette` (str): Seaborn color palette name
- `style` (str): Plot style ('seaborn', 'matplotlib', etc.)

---

### `plotsense.explainer()`

Generates natural language explanations for visualizations.

**Signature:**
```python
plotsense.explainer(plot, custom_prompt=None, iterations=1)
```

**Parameters:**
- `plot` (matplotlib.figure.Figure): The plot to explain (from `ps.plotgen()`)
- `custom_prompt` (str, optional): Custom prompt to guide the explanation
- `iterations` (int, optional): Number of explanation refinements (default: 1)

**Returns:**
- `str`: Natural language explanation of the visualization

**Example:**
```python
from plotsense import explainer

# Generate a plot
plot = plotgen(df, suggestion)

# Get basic explanation
explanation = explainer(plot)
print(explanation)

# Get custom explanation
custom_explanation = explainer(
    plot,
    custom_prompt="Focus on outliers and trends",
    iterations=2
)
print(custom_explanation)
```

**Custom Prompt Examples:**
```python
# Focus on specific aspects
explainer(plot, "Explain the correlation between variables")
explainer(plot, "Identify any outliers or anomalies")
explainer(plot, "Describe the distribution pattern")
explainer(plot, "Explain this chart for a business audience")
```

---

## Utility Functions

### `plotsense.get_version()`

Returns the current version of PlotSense.

**Signature:**
```python
plotsense.get_version()
```

**Returns:**
- `str`: Version string

**Example:**
```python
import plotsense as ps
print(ps.get_version())  # e.g., "1.0.0"
```

---

### `plotsense.list_supported_plots()`

Returns a list of all supported plot types.

**Signature:**
```python
plotsense.list_supported_plots()
```

**Returns:**
- `list`: List of supported plot type strings

**Example:**
```python
import plotsense as ps
supported_plots = ps.list_supported_plots()
print(supported_plots)
# ['scatter', 'bar', 'barh', 'histogram', 'boxplot', 'violinplot', 'pie', 'hexbin']
```

---

## Configuration Functions

### `plotsense.set_api_key()`

Sets the Groq API key programmatically.

**Signature:**
```python
plotsense.set_api_key(api_key)
```

**Parameters:**
- `api_key` (str): Your Groq API key

**Example:**
```python
import plotsense as ps
ps.set_api_key("your-groq-api-key-here")
```

---

### `plotsense.configure()`

Configure global PlotSense settings.

**Signature:**
```python
plotsense.configure(**kwargs)
```

**Parameters:**
- `default_figsize` (tuple): Default figure size for all plots
- `default_style` (str): Default plot style
- `max_recommendations` (int): Maximum number of recommendations to return
- `timeout` (int): API timeout in seconds
- `cache_recommendations` (bool): Whether to cache recommendations

**Example:**
```python
import plotsense as ps

ps.configure(
    default_figsize=(12, 8),
    default_style='seaborn-v0_8',
    max_recommendations=5,
    timeout=30
)
```

---

## Error Handling

PlotSense raises specific exceptions for different error conditions:

### `PlotSenseError`
Base exception class for all PlotSense errors.

### `APIError`
Raised when there are issues with the Groq API (network, authentication, etc.).

### `DataError`
Raised when there are issues with the input data format or content.

### `PlotGenerationError`
Raised when plot generation fails.

**Example Error Handling:**
```python
import plotsense as ps
from plotsense.exceptions import APIError, DataError

try:
    suggestions = ps.recommender(df)
    plot = ps.plotgen(df, suggestions.iloc[0])
except APIError as e:
    print(f"API Error: {e}")
except DataError as e:
    print(f"Data Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Data Requirements

### DataFrame Structure
- Must be a valid pandas DataFrame
- At least 2 rows of data
- At least 1 column with meaningful data
- Column names should be descriptive

### Supported Data Types
- **Numerical**: int, float, numpy numeric types
- **Categorical**: string, category, object
- **Temporal**: datetime, date, timestamp
- **Boolean**: bool

### Data Quality Guidelines
- Handle missing values before passing to PlotSense
- Ensure categorical variables have reasonable number of categories (< 50)
- Large datasets (> 10,000 rows) may have slower performance
- Column names should not contain special characters or start with numbers

---

## Performance Considerations

### Optimization Tips
1. **Data Size**: Optimal performance with 100-10,000 rows
2. **API Calls**: Recommendations and explanations require internet connectivity
3. **Caching**: Enable recommendation caching for repeated analysis
4. **Data Preprocessing**: Clean data beforehand for faster processing

### Rate Limits
- Groq API has rate limits based on your plan
- PlotSense automatically handles retries with exponential backoff
- Consider upgrading your Groq plan for heavy usage

---

## Migration Guide

### From Version 0.x to 1.x
- Function signatures remain the same
- New optional parameters added to existing functions
- Enhanced error handling and validation
- Improved performance and reliability

### Deprecated Features
- None in current version

---

## See Also

- [Getting Started](getting-started.md) - Basic usage examples
- [Examples](examples.md) - Comprehensive examples
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Configuration](configuration.md) - Advanced configuration options
