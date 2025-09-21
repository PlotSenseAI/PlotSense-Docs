# Configuration Guide

This guide covers how to configure PlotSense for optimal performance and customization.

## API Configuration

### Setting Up Your Groq API Key

PlotSense requires a Groq API key for AI-powered features. Here are all the ways to configure it:

#### Method 1: Environment Variable (Recommended)

**Windows:**
```cmd
# Temporary (current session only)
set GROQ_API_KEY=your-api-key-here

# Permanent (add to system environment variables)
setx GROQ_API_KEY "your-api-key-here"
```

**macOS/Linux:**
```bash
# Temporary (current session only)
export GROQ_API_KEY="your-api-key-here"

# Permanent (add to shell profile)
echo 'export GROQ_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# For zsh users
echo 'export GROQ_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

#### Method 2: Configuration File

Create a `.env` file in your project directory:
```bash
# .env file
GROQ_API_KEY=your-api-key-here
PLOTSENSE_CACHE_DIR=/path/to/cache
PLOTSENSE_DEFAULT_STYLE=seaborn-v0_8
```

Load it in your Python script:
```python
from dotenv import load_dotenv
import plotsense as ps

load_dotenv()  # This loads the .env file
```

#### Method 3: Programmatic Configuration

```python
import plotsense as ps
import os

# Set API key directly
os.environ["GROQ_API_KEY"] = "your-api-key-here"

# Or use PlotSense's configuration function
ps.set_api_key("your-api-key-here")
```

#### Method 4: Runtime Configuration

```python
import plotsense as ps
from plotsense import recommender
from plotsense import plotgen
from plotsense import explainer

# Configure at runtime with context manager
with ps.api_key_context("your-api-key-here"):
    suggestions = recommender(df)
    plot = plotgen(df, suggestions.iloc[0])
```

## Global Configuration

### Using `ps.configure()`

Set global defaults for all PlotSense operations:

```python
import plotsense as ps

ps.configure(
    # API settings
    api_timeout=30,                    # Timeout in seconds
    max_retries=3,                     # Number of API retry attempts

    # Visualization settings
    default_figsize=(12, 8),           # Default figure size
    default_style='seaborn-v0_8',      # Default plot style
    default_dpi=100,                   # Default DPI for plots

    # Recommendation settings
    max_recommendations=10,            # Max number of suggestions
    min_data_points=5,                 # Minimum rows required
    cache_recommendations=True,        # Enable caching

    # Output settings
    verbose=True,                      # Enable verbose output
    show_warnings=True,                # Show warning messages
    auto_save_plots=False,             # Auto-save generated plots
    save_directory="./plots/"          # Directory for auto-saved plots
)
```

### Configuration Options Reference

#### API Settings

```python
ps.configure(
    api_timeout=30,              # Request timeout (seconds)
    api_base_url="custom-url",   # Custom API endpoint
    max_retries=3,               # Retry attempts for failed requests
    retry_delay=1.0,             # Delay between retries (seconds)
    rate_limit_delay=0.1         # Delay between requests (seconds)
)
```

#### Visualization Settings

```python
ps.configure(
    # Figure settings
    default_figsize=(10, 6),         # Width, height in inches
    default_dpi=100,                 # Dots per inch
    default_style='seaborn-v0_8',    # Matplotlib style

    # Color settings
    default_palette='viridis',       # Default color palette
    color_blind_friendly=True,       # Use color-blind friendly palettes

    # Font settings
    font_family='Arial',             # Default font family
    font_size=12,                    # Default font size
    title_font_size=14,              # Title font size

    # Layout settings
    tight_layout=True,               # Use tight layout
    grid=True,                       # Show grid by default
    legend=True                      # Show legend by default
)
```

#### Performance Settings

```python
ps.configure(
    # Caching
    cache_recommendations=True,       # Cache AI recommendations
    cache_directory="~/.plotsense",   # Cache location
    cache_max_size=100,              # Max cached items
    cache_ttl=3600,                  # Cache time-to-live (seconds)

    # Data processing
    max_data_points=10000,           # Max rows to process
    sampling_strategy='random',       # 'random', 'systematic', 'none'
    sample_size=5000,                # Sample size for large datasets

    # Memory management
    cleanup_plots=True,              # Auto-cleanup plot objects
    memory_limit_mb=512              # Memory limit for processing
)
```

#### Output Settings

```python
ps.configure(
    # Logging
    verbose=True,                    # Verbose output
    log_level='INFO',                # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_file='plotsense.log',        # Log file path

    # Auto-save
    auto_save_plots=False,           # Automatically save plots
    save_directory='./plots/',       # Save directory
    save_format='png',               # 'png', 'pdf', 'svg', 'jpg'
    save_dpi=300,                    # DPI for saved plots

    # Display
    show_plots=True,                 # Automatically display plots
    interactive_mode=False           # Enable interactive features
)
```

## Environment-Specific Configuration

### Development Environment

```python
import plotsense as ps

# Development configuration
ps.configure(
    verbose=True,
    show_warnings=True,
    cache_recommendations=False,  # Disable cache for testing
    auto_save_plots=True,
    save_directory='./dev_plots/',
    log_level='DEBUG'
)
```

### Production Environment

```python
import plotsense as ps

# Production configuration
ps.configure(
    verbose=False,
    show_warnings=False,
    cache_recommendations=True,
    auto_save_plots=False,
    log_level='ERROR',
    api_timeout=10,              # Shorter timeout for production
    max_retries=1                # Fewer retries for faster responses
)
```

### Jupyter Notebook Environment

```python
import plotsense as ps

# Jupyter-optimized configuration
ps.configure(
    default_figsize=(10, 6),
    default_dpi=100,             # Lower DPI for faster rendering
    show_plots=True,
    interactive_mode=True,
    tight_layout=True,
    verbose=False                # Reduce output clutter
)
```

## Advanced Configuration

### Custom Model Configuration

```python
import plotsense as ps

# Configure AI model settings
ps.configure_model(
    model_name='llama-3.1-70b-versatile',  # Groq model to use
    temperature=0.1,                        # Model creativity (0-1)
    max_tokens=1000,                        # Maximum response length
    top_p=0.9,                             # Nucleus sampling parameter
    custom_system_prompt="Custom instructions for the AI..."
)
```

### Custom Plot Templates

```python
import plotsense as ps

# Define custom plot templates
custom_templates = {
    'business_report': {
        'figsize': (12, 8),
        'style': 'seaborn-v0_8-whitegrid',
        'palette': 'Set2',
        'title_fontsize': 16,
        'label_fontsize': 12,
        'grid': True,
        'legend': True
    },
    'scientific_paper': {
        'figsize': (8, 6),
        'style': 'classic',
        'palette': 'gray',
        'title_fontsize': 14,
        'label_fontsize': 10,
        'grid': False,
        'legend': False
    }
}

# Register templates
ps.register_templates(custom_templates)

# Use template
plot = ps.plotgen(df, suggestion, template='business_report')
```

### Custom Explanation Prompts

```python
import plotsense as ps

# Define domain-specific explanation prompts
explanation_prompts = {
    'business': "Explain this visualization from a business strategy perspective, focusing on actionable insights and KPIs.",
    'scientific': "Provide a scientific analysis of the patterns, including statistical significance and methodology considerations.",
    'marketing': "Analyze this data from a marketing perspective, highlighting customer segments and campaign opportunities.",
    'financial': "Provide financial analysis including risk assessment, trends, and investment implications."
}

# Register prompts
ps.register_explanation_prompts(explanation_prompts)

# Use custom prompt
explanation = ps.explainer(plot, prompt_type='business')
```

## Configuration Files

### YAML Configuration

Create `plotsense_config.yaml`:

```yaml
# plotsense_config.yaml
api:
  timeout: 30
  max_retries: 3
  rate_limit_delay: 0.1

visualization:
  default_figsize: [12, 8]
  default_style: "seaborn-v0_8"
  default_palette: "viridis"
  default_dpi: 100

performance:
  cache_recommendations: true
  cache_directory: "~/.plotsense"
  max_data_points: 10000
  sampling_strategy: "random"

output:
  verbose: false
  auto_save_plots: false
  save_format: "png"
  log_level: "INFO"
```

Load YAML configuration:

```python
import plotsense as ps
import yaml

# Load configuration from YAML
with open('plotsense_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

ps.configure(**config['api'])
ps.configure(**config['visualization'])
ps.configure(**config['performance'])
ps.configure(**config['output'])
```

### JSON Configuration

Create `plotsense_config.json`:

```json
{
  "api": {
    "timeout": 30,
    "max_retries": 3
  },
  "visualization": {
    "default_figsize": [12, 8],
    "default_style": "seaborn-v0_8",
    "default_palette": "viridis"
  },
  "performance": {
    "cache_recommendations": true,
    "max_data_points": 10000
  }
}
```

Load JSON configuration:

```python
import plotsense as ps
import json

# Load configuration from JSON
with open('plotsense_config.json', 'r') as file:
    config = json.load(file)

for section, settings in config.items():
    ps.configure(**settings)
```

## Validation and Testing

### Validate Configuration

```python
import plotsense as ps

# Test configuration
config_status = ps.validate_config()
print(f"Configuration valid: {config_status['valid']}")

if not config_status['valid']:
    print("Issues found:")
    for issue in config_status['issues']:
        print(f"- {issue}")
```

### Test API Connection

```python
import plotsense as ps

# Test API connectivity
api_status = ps.test_api_connection()
print(f"API accessible: {api_status['connected']}")
print(f"Response time: {api_status['response_time']}ms")

if not api_status['connected']:
    print(f"Error: {api_status['error']}")
```

## Troubleshooting Configuration

### Common Configuration Issues

1. **API Key Not Found**
   ```python
   # Check if API key is set
   import os
   api_key = os.getenv('GROQ_API_KEY')
   if not api_key:
       print("API key not found. Please set GROQ_API_KEY environment variable.")
   ```

2. **Invalid Configuration Values**
   ```python
   # Validate configuration values
   try:
       ps.configure(default_figsize=(10, 'invalid'))
   except ValueError as e:
       print(f"Configuration error: {e}")
   ```

3. **Cache Directory Issues**
   ```python
   # Check cache directory
   import os
   cache_dir = os.path.expanduser('~/.plotsense')
   if not os.path.exists(cache_dir):
       os.makedirs(cache_dir)
       print(f"Created cache directory: {cache_dir}")
   ```

### Reset to Defaults

```python
import plotsense as ps

# Reset all configuration to defaults
ps.reset_config()

# Reset specific sections
ps.reset_config(sections=['api', 'visualization'])

# Get current configuration
current_config = ps.get_config()
print(current_config)
```

## Next Steps

- Review [Troubleshooting Guide](troubleshooting.md) for common issues
- Check [Examples](examples.md) for configuration in practice
- See [API Reference](api-reference.md) for detailed parameter information
- Visit [GitHub](https://github.com/PlotSenseAI/PlotSense) for the latest updates
