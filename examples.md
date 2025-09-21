# Examples and Use Cases

This page provides comprehensive examples of using PlotSense for various data visualization scenarios.

## Basic Examples

### Example 1: Sales Data Analysis

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd

# Create sample sales data
sales_data = {
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'revenue': [45000, 52000, 48000, 61000, 55000, 67000],
    'expenses': [32000, 35000, 33000, 42000, 38000, 45000],
    'region': ['North', 'North', 'South', 'South', 'North', 'South']
}
df = pd.DataFrame(sales_data)

# Get AI recommendations
suggestions = recommender(df)
print("Top 3 recommendations:")
for i in range(min(3, len(suggestions))):
    print(f"{i+1}. {suggestions.iloc[i]['description']}")

# Generate the top recommendation
plot = plotgen(df, suggestions.iloc[0])
plot.show()

# Get AI explanation
explanation = explainer(plot)
print(f"\\nInsight: {explanation}")
```

### Example 2: Customer Demographics

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import numpy as np

# Generate customer data
np.random.seed(42)
customers = pd.DataFrame({
    'age': np.random.normal(35, 12, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'satisfaction': np.random.uniform(1, 10, 1000),
    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000),
    'months_active': np.random.randint(1, 36, 1000)
})

# Clean negative values
customers['age'] = customers['age'].clip(18, 80)
customers['income'] = customers['income'].clip(20000, 200000)

# Get recommendations
suggestions = recommender(customers)

# Generate multiple plots
for i in range(3):
    print(f"\\n=== Visualization {i+1} ===")
    plot = plotgen(customers, suggestions.iloc[i])

    # Custom explanation focusing on business insights
    explanation = explainer(
        plot,
        custom_prompt="Explain this from a business strategy perspective"
    )
    print(f"Business Insight: {explanation}")
```

## Advanced Examples

### Example 3: Time Series Analysis

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(123)

# Simulate website traffic with trend and seasonality
trend = np.linspace(1000, 2000, 365)
seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly pattern
noise = np.random.normal(0, 100, 365)
traffic = trend + seasonal + noise

website_data = pd.DataFrame({
    'date': dates,
    'daily_visitors': traffic.astype(int),
    'page_views': (traffic * np.random.uniform(2.5, 4.0, 365)).astype(int),
    'bounce_rate': np.random.uniform(0.3, 0.7, 365),
    'conversion_rate': np.random.uniform(0.02, 0.08, 365),
    'day_of_week': dates.day_name(),
    'month': dates.month_name()
})

# Get recommendations for time series
suggestions = recommender(website_data)

# Generate temporal visualization
plot = plotgen(website_data, suggestions.iloc[0])

# Get detailed explanation with multiple iterations
explanation = explainer(
    plot,
    custom_prompt="Analyze trends, patterns, and anomalies in this time series",
    iterations=2
)
print(f"Time Series Analysis: {explanation}")
```

### Example 4: Multi-Dataset Comparison

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd

# Compare different product categories
products_q1 = pd.DataFrame({
    'category': ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'],
    'sales_q1': [125000, 89000, 45000, 67000, 34000],
    'units_sold_q1': [450, 890, 1200, 340, 280],
    'quarter': 'Q1'
})

products_q2 = pd.DataFrame({
    'category': ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'],
    'sales_q2': [142000, 95000, 41000, 72000, 38000],
    'units_sold_q2': [520, 950, 1100, 380, 310],
    'quarter': 'Q2'
})

# Combine datasets for comparison
combined_data = pd.DataFrame({
    'category': products_q1['category'].tolist() + products_q2['category'].tolist(),
    'sales': products_q1['sales_q1'].tolist() + products_q2['sales_q2'].tolist(),
    'units_sold': products_q1['units_sold_q1'].tolist() + products_q2['units_sold_q2'].tolist(),
    'quarter': ['Q1'] * 5 + ['Q2'] * 5
})

# Get recommendations
suggestions = recommender(combined_data)

# Generate comparison plot
plot = plotgen(combined_data, suggestions.iloc[0])

# Get comparative analysis
explanation = explainer(
    plot,
    custom_prompt="Compare performance between quarters and identify winners/losers"
)
print(f"Quarterly Comparison: {explanation}")
```

## Domain-Specific Examples

### Example 5: Scientific Data Visualization

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import numpy as np

# Simulate experimental data
np.random.seed(456)
experiments = pd.DataFrame({
    'temperature': np.random.uniform(20, 100, 200),
    'pressure': np.random.uniform(1, 10, 200),
    'reaction_rate': np.random.gamma(2, 2, 200),
    'catalyst_type': np.random.choice(['A', 'B', 'C'], 200),
    'ph_level': np.random.uniform(6, 8, 200),
    'yield_percentage': np.random.beta(8, 2, 200) * 100
})

# Add some correlation
experiments['reaction_rate'] = (
    experiments['temperature'] * 0.05 +
    experiments['pressure'] * 0.3 +
    np.random.normal(0, 0.5, 200)
)

# Get scientific visualization recommendations
suggestions = recommender(experiments)

# Generate scientific plot
plot = plotgen(experiments, suggestions.iloc[0])

# Get scientific explanation
explanation = explainer(
    plot,
    custom_prompt="Explain the relationships and patterns from a scientific research perspective"
)
print(f"Scientific Analysis: {explanation}")
```

### Example 6: Financial Data Analysis

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import numpy as np

# Simulate stock portfolio data
np.random.seed(789)
portfolio = pd.DataFrame({
    'stock_symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'] * 60,
    'date': pd.date_range('2023-01-01', periods=300, freq='D'),
    'price': np.random.uniform(50, 500, 300),
    'volume': np.random.randint(1000000, 50000000, 300),
    'market_cap': np.random.uniform(1e9, 3e12, 300),
    'pe_ratio': np.random.uniform(10, 50, 300),
    'sector': np.random.choice(['Tech', 'Consumer', 'Energy'], 300)
})

# Add realistic price movements
for symbol in portfolio['stock_symbol'].unique():
    mask = portfolio['stock_symbol'] == symbol
    base_price = np.random.uniform(100, 400)
    returns = np.random.normal(0.001, 0.02, mask.sum())
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    portfolio.loc[mask, 'price'] = prices

# Get financial recommendations
suggestions = recommender(portfolio)

# Generate financial visualization
plot = plotgen(portfolio, suggestions.iloc[0])

# Get financial analysis
explanation = explainer(
    plot,
    custom_prompt="Provide investment insights and risk analysis based on this data"
)
print(f"Investment Analysis: {explanation}")
```

## Customization Examples

### Example 7: Custom Plot Styling

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd

# Sample data
df = pd.DataFrame({
    'x': range(10),
    'y': [i**2 for i in range(10)],
    'category': ['A', 'B'] * 5
})

# Get recommendations
suggestions = recommender(df)

# Generate plot with custom styling
plot = plotgen(
    df,
    suggestions.iloc[0],
    figsize=(12, 8),
    title="Custom Styled Visualization",
    xlabel="Custom X Label",
    ylabel="Custom Y Label",
    color_palette="viridis",
    style="seaborn-v0_8"
)

# Save the plot
plot.savefig("custom_plot.png", dpi=300, bbox_inches='tight')
```

### Example 8: Custom Recommendations

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd

# Sample data
df = pd.DataFrame({
    'height': [160, 165, 170, 175, 180, 185],
    'weight': [55, 60, 65, 70, 75, 80],
    'age': [25, 30, 35, 40, 45, 50],
    'gender': ['F', 'F', 'M', 'M', 'M', 'F']
})

# Create custom recommendation
custom_rec = {
    'plot_type': 'scatter',
    'x_column': 'height',
    'y_column': 'weight',
    'color_column': 'gender',
    'description': 'Height vs Weight by Gender'
}

# Generate plot from custom recommendation
plot = plotgen(df, custom_rec)

# Get explanation
explanation = explainer(plot, "Analyze the relationship between physical attributes")
print(f"Custom Analysis: {explanation}")
```

## Interactive Examples

### Example 9: Multiple Plot Comparison

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data
df = pd.DataFrame({
    'A': np.random.normal(0, 1, 100),
    'B': np.random.normal(2, 1.5, 100),
    'C': np.random.exponential(1, 100),
    'category': np.random.choice(['X', 'Y', 'Z'], 100)
})

# Get multiple recommendations
suggestions = recommender(df)

# Create subplot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Multiple Visualization Comparison', fontsize=16)

for i in range(min(4, len(suggestions))):
    row, col = i // 2, i % 2

    # Generate individual plots
    plot = plotgen(df, suggestions.iloc[i])

    # Copy to subplot (simplified - actual implementation may vary)
    axes[row, col].set_title(f"Recommendation {i+1}: {suggestions.iloc[i]['description']}")

    # Get explanation for each
    explanation = explainer(plot)
    print(f"Plot {i+1}: {explanation[:100]}...")

plt.tight_layout()
plt.show()
```

### Example 10: Real-time Data Visualization

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import time

def simulate_real_time_data():
    """Simulate streaming data"""
    base_time = pd.Timestamp.now()

    for i in range(10):
        # Generate new data point
        new_data = pd.DataFrame({
            'timestamp': [base_time + pd.Timedelta(seconds=i*5)],
            'sensor_1': [np.random.normal(25, 5)],
            'sensor_2': [np.random.normal(50, 10)],
            'alert_level': [np.random.choice(['Low', 'Medium', 'High'])]
        })

        # Accumulate data
        if i == 0:
            streaming_data = new_data
        else:
            streaming_data = pd.concat([streaming_data, new_data], ignore_index=True)

        print(f"\\nData point {i+1} received...")

        # Get recommendations for current data
        if len(streaming_data) >= 3:  # Need minimum data for recommendations
            suggestions = recommender(streaming_data)
            plot = plotgen(streaming_data, suggestions.iloc[0])

            # Quick analysis
            explanation = explainer(
                plot,
                "Provide real-time monitoring insights and any alerts"
            )
            print(f"Real-time Insight: {explanation}")

        time.sleep(1)  # Simulate delay

# Run simulation
# simulate_real_time_data()
```

## Best Practices Examples

### Example 11: Data Quality Checks

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import numpy as np

def analyze_with_quality_checks(df):
    """Analyze data with proper quality checks"""

    print("=== Data Quality Report ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")

    # Check for minimum requirements
    if len(df) < 2:
        print("Error: Dataset too small for visualization")
        return

    if len(df.columns) < 1:
        print("Error: No columns available for visualization")
        return

    # Clean data
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")

    # Get recommendations
    try:
        suggestions = recommender(df_clean)

        if len(suggestions) > 0:
            print(f"\\n=== Generated {len(suggestions)} recommendations ===")

            # Generate best visualization
            plot = plotgen(df_clean, suggestions.iloc[0])

            # Get comprehensive explanation
            explanation = explainer(
                plot,
                "Provide comprehensive analysis including data quality observations"
            )
            print(f"\\nComprehensive Analysis: {explanation}")

        else:
            print("No suitable visualizations found for this dataset")

    except Exception as e:
        print(f"Error during analysis: {e}")

# Example usage
sample_data = pd.DataFrame({
    'numeric_col': [1, 2, 3, np.nan, 5],
    'category_col': ['A', 'B', 'A', 'C', 'B'],
    'text_col': ['hello', 'world', 'test', 'data', 'viz']
})

analyze_with_quality_checks(sample_data)
```

## Integration Examples

### Example 12: Jupyter Notebook Integration

```python
# In Jupyter Notebook
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
from IPython.display import display, HTML

# Enable inline plotting
%matplotlib inline

# Load data
df = pd.read_csv("your_data.csv")

# Display data info
display(HTML("<h3>Dataset Overview</h3>"))
display(df.head())
display(df.describe())

# Get and display recommendations
suggestions = recommender(df)
display(HTML("<h3>AI Recommendations</h3>"))
display(suggestions)

# Generate interactive plot selection
for i, suggestion in suggestions.iterrows():
    plot = plotgen(df, suggestion)
    display(HTML(f"<h4>Recommendation {i+1}: {suggestion['description']}</h4>"))
    display(plot)

    explanation = explainer(plot)
    display(HTML(f"<p><strong>Analysis:</strong> {explanation}</p>"))
    display(HTML("<hr>"))
```

### Example 13: Web Application Integration

```python
# Flask web application example
from flask import Flask, render_template, request, jsonify
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer
import pandas as pd
import io
import base64

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        # Get uploaded data
        file = request.files['data_file']
        df = pd.read_csv(file)

        # Get recommendations
        suggestions = recommender(df)

        # Generate plot
        plot = plotgen(df, suggestions.iloc[0])

        # Convert plot to base64 for web display
        img_buffer = io.BytesIO()
        plot.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()

        # Get explanation
        explanation = explainer(plot)

        return jsonify({
            'success': True,
            'plot_image': f"data:image/png;base64,{img_str}",
            'explanation': explanation,
            'suggestions_count': len(suggestions)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
```

## Next Steps

After exploring these examples:

1. **Experiment** with your own datasets using similar patterns
2. **Customize** the visualizations based on your specific needs
3. **Integrate** PlotSense into your existing data analysis workflow
4. **Check** the [API Reference](api-reference.md) for detailed parameter options
5. **Review** [Configuration](configuration.md) for advanced settings

For more specific use cases or questions, visit our [GitHub repository](https://github.com/PlotSenseAI/PlotSense) or check the [Troubleshooting Guide](troubleshooting.md).
