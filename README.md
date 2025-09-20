# PlotSense Documentation

Welcome to the official documentation for PlotSense - an AI-powered data visualization assistant that helps data professionals and analysts create smarter, faster, and more explainable visualizations.

## Quick Links

- [Getting Started](getting-started.md)
- [Installation](installation.md)
- [API Reference](api-reference.md)
- [Examples](examples.md)
- [Configuration](configuration.md)
- [Troubleshooting](troubleshooting.md)
- [Contributing](contributing.md)

## What is PlotSense?

PlotSense is an AI-powered assistant that transforms how you approach data visualization. It analyzes your datasets, recommends optimal chart types, generates visualizations with a single command, and provides intelligent explanations of your plots.

### Key Features

- **AI Visualization Recommendations**: Get intelligent suggestions for the best chart types based on your data structure
- **One-Click Plot Generation**: Create beautiful visualizations instantly from recommendations
- **AI-Powered Explanations**: Understand your data through natural language insights
- **Multiple Plot Types**: Support for scatter, bar, histogram, boxplot, pie charts, and more
- **Pandas Integration**: Seamless workflow with your existing data analysis pipeline

## Quick Start

```python
import plotsense as ps
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Get AI recommendations
suggestions = ps.recommender(df)

# Generate a plot
plot = ps.plotgen(df, suggestions.iloc[0])

# Get AI explanation
explanation = ps.explainer(plot)
print(explanation)
```

## Support

If you encounter any issues or have questions, please:
- Check our [Troubleshooting Guide](troubleshooting.md)
- Visit our [GitHub repository](https://github.com/PlotSenseAI/PlotSense)
- Submit an issue on GitHub

## License

PlotSense is licensed under the Apache License 2.0.