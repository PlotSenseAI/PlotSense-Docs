# Contributing to PlotSense

We welcome contributions to PlotSense! This guide will help you get started with contributing to the project.

## Getting Started

### Prerequisites

Before contributing, ensure you have:
- Python 3.7 or higher
- Git installed and configured
- A GitHub account
- Basic knowledge of Python, pandas, and matplotlib

### Development Setup

1. **Fork the Repository**
   ```bash
   # Visit https://github.com/PlotSenseAI/PlotSense and click "Fork"
   ```

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PlotSense.git
   cd PlotSense
   ```

3. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv plotsense-dev
   source plotsense-dev/bin/activate  # On Windows: plotsense-dev\Scripts\activate

   # Install dependencies
   pip install -e ".[dev]"  # Install in development mode

   # Install pre-commit hooks
   pre-commit install

   # import modules
   from plotsense import recommender
   from plotsense import plotgen
   from plotsense import explainer
   ```

4. **Set Up API Key**
   ```bash
   # Create .env file in project root
   echo "GROQ_API_KEY=your-api-key-here" > .env
   ```

5. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   python -m pytest tests/
   ```

## Types of Contributions

### 1. Bug Reports

Found a bug? Help us fix it!

**Before reporting:**
- Check existing issues to avoid duplicates
- Test with the latest version
- Create a minimal reproducible example

**Bug Report Template:**
```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Import PlotSense
2. Load data: `df = pd.read_csv('example.csv')`
3. Call function: `recommender(df)`
4. See error

**Expected Behavior**
What you expected to happen

**Environment**
- Python version:
- PlotSense version:
- OS:
- Dependencies versions:

**Additional Context**
Screenshots, error logs, etc.
```

### 2. Feature Requests

Have an idea for improvement?

**Feature Request Template:**
```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Explain why this feature would be useful

**Proposed Solution**
Your ideas for implementation

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Examples, mockups, references
```

### 3. Code Contributions

Ready to contribute code? Great!

#### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make Changes**
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   python -m pytest

   # Run specific test
   python -m pytest tests/test_recommender.py

   # Run with coverage
   python -m pytest --cov=plotsense

   # Run linting
   flake8 plotsense/
   black plotsense/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new visualization type support

   - Add support for violin plots
   - Update recommender logic for violin plot suggestions
   - Add tests for violin plot generation
   - Update documentation"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub
   ```

## Code Guidelines

### Code Style

We follow PEP 8 with some modifications:

```python
# Good examples
def recommend_visualizations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate visualization recommendations for a dataset.

    Args:
        dataframe: Input pandas DataFrame to analyze

    Returns:
        DataFrame containing visualization recommendations
    """
    if dataframe.empty:
        raise ValueError("DataFrame cannot be empty")

    # Process dataframe
    suggestions = _analyze_data_structure(dataframe)
    return _format_recommendations(suggestions)


class PlotGenerator:
    """Handles plot generation from recommendations."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def generate_plot(self, data: pd.DataFrame, suggestion: Dict) -> Figure:
        """Generate plot from data and suggestion."""
        plot_type = suggestion.get('plot_type')

        if plot_type == 'scatter':
            return self._create_scatter_plot(data, suggestion)
        elif plot_type == 'bar':
            return self._create_bar_plot(data, suggestion)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
```

### Docstring Format

Use Google-style docstrings:

```python
def explainer(plot: Figure, custom_prompt: Optional[str] = None) -> str:
    """Generate natural language explanation for a plot.

    This function analyzes a matplotlib Figure and generates a human-readable
    explanation of the visualization using AI.

    Args:
        plot: The matplotlib Figure to explain
        custom_prompt: Optional custom prompt to guide explanation generation.
            If not provided, uses default explanation prompt.

    Returns:
        Natural language explanation of the plot as a string.

    Raises:
        APIError: If the AI service is unavailable or returns an error
        ValueError: If the plot parameter is invalid

    Example:
        >>> plot = plotgen(df, suggestion)
        >>> explanation = explainer(plot)
        >>> print(explanation)
        "This scatter plot shows a positive correlation between x and y..."
    """
```

### Testing Guidelines

Write comprehensive tests for all new functionality:

```python
import pytest
import pandas as pd
import numpy as np
from plotsense import recommender, plotgen, explainer


class TestRecommender:
    """Test suite for recommendation functionality."""

    def setup_method(self):
        """Set up test data."""
        self.simple_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],
            'category': ['A', 'B', 'A', 'B', 'A']
        })

    def test_recommender_basic_functionality(self):
        """Test basic recommender functionality."""
        suggestions = recommender(self.simple_df)

        assert isinstance(suggestions, pd.DataFrame)
        assert len(suggestions) > 0
        assert 'plot_type' in suggestions.columns
        assert 'description' in suggestions.columns

    def test_recommender_empty_dataframe(self):
        """Test recommender with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            recommender(empty_df)

    def test_recommender_numeric_data(self):
        """Test recommendations for numeric data."""
        numeric_df = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.normal(0, 1, 100)
        })

        suggestions = recommender(numeric_df)
        plot_types = suggestions['plot_type'].unique()

        assert 'scatter' in plot_types
        assert len(suggestions) > 0

    @pytest.mark.parametrize("plot_type", ['scatter', 'bar', 'histogram'])
    def test_plotgen_supported_types(self, plot_type):
        """Test plot generation for supported types."""
        suggestion = {
            'plot_type': plot_type,
            'x_column': 'x',
            'y_column': 'y' if plot_type != 'histogram' else None
        }

        plot = plotgen(self.simple_df, suggestion)
        assert plot is not None
        assert hasattr(plot, 'savefig')  # Check it's a matplotlib Figure
```

### Performance Considerations

- Write efficient code that scales with data size
- Use vectorized operations where possible
- Add performance tests for critical functions
- Profile code for bottlenecks

```python
def test_performance_large_dataset():
    """Test performance with large datasets."""
    import time

    # Create large dataset
    large_df = pd.DataFrame({
        'x': np.random.random(10000),
        'y': np.random.random(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })

    start_time = time.time()
    suggestions = recommender(large_df)
    duration = time.time() - start_time

    assert duration < 5.0  # Should complete within 5 seconds
    assert len(suggestions) > 0
```

## Documentation Contributions

### Updating Documentation

1. **API Documentation**: Update docstrings for any new or modified functions
2. **User Guides**: Update relevant guides in the docs folder
3. **Examples**: Add examples for new features
4. **README**: Update if adding major features

### Documentation Style

- Use clear, concise language
- Include practical examples
- Keep code examples up-to-date
- Use consistent formatting

```markdown
## Function Name

Brief description of what the function does.

### Parameters

- `param1` (type): Description of parameter
- `param2` (type, optional): Description with default behavior

### Returns

- `type`: Description of return value

### Example

```python
import plotsense as ps
from plotsense import function_name

# Simple example
result = function_name(data)
print(result)
```

### Notes

Any important notes or limitations.
```

## Release Process

### Version Numbering

We use Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- Major: Breaking changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Preparing a Release

1. **Update Version Numbers**
   ```python
   # In setup.py and __init__.py
   __version__ = "1.2.3"
   ```

2. **Update Changelog**
   ```markdown
   ## [1.2.3] - 2024-01-15

   ### Added
   - New visualization type: violin plots
   - Support for custom color palettes

   ### Fixed
   - Bug in recommendation scoring
   - Memory leak in plot generation

   ### Changed
   - Improved API response handling
   ```

3. **Run Full Test Suite**
   ```bash
   python -m pytest tests/ --cov=plotsense
   flake8 plotsense/
   black --check plotsense/
   ```

4. **Create Release PR**
   - Include all changes since last release
   - Update documentation
   - Tag the release

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions, documentation updates
- **Discussions**: General questions, ideas, community support

### Review Process

All contributions go through review:

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Peer Review**: At least one maintainer reviews code
3. **Documentation Review**: Ensure docs are updated
4. **Final Approval**: Maintainer approves and merges

## Getting Help

### Development Questions

- Check existing issues and discussions
- Ask in GitHub Discussions
- Join community channels (if available)

### Setting Up Development Environment

If you encounter issues:

1. **Check Prerequisites**: Ensure Python version, dependencies
2. **Virtual Environment**: Use clean virtual environment
3. **API Keys**: Ensure test API keys are configured
4. **Platform Issues**: Check platform-specific requirements

### Common Development Issues

```python
# Issue: Tests failing locally
# Solution: Ensure test dependencies are installed
pip install -e ".[test]"

# Issue: Import errors
# Solution: Install in development mode
pip install -e .

# Issue: API tests failing
# Solution: Check API key configuration
echo $GROQ_API_KEY  # Should show your key
```

## Recognition

Contributors are recognized in:
- Release notes
- Contributors section in README
- GitHub contributors page
- Special mentions for significant contributions

Thank you for contributing to PlotSense! Your contributions make the project better for everyone.

## Quick Reference

### Essential Commands

```bash
# Setup
git clone https://github.com/YOUR_USERNAME/PlotSense.git
cd PlotSense
python -m venv plotsense-dev
source plotsense-dev/bin/activate
pip install -e ".[dev]"

# Development
git checkout -b feature/new-feature
# Make changes
python -m pytest
git commit -m "feat: description"
git push origin feature/new-feature

# Before submitting PR
python -m pytest --cov=plotsense
flake8 plotsense/
black plotsense/
```

### File Structure

```
PlotSense/
├── plotsense/           # Main package
│   ├── __init__.py
│   ├── recommender.py   # AI recommendation logic
│   ├── plotgen.py       # Plot generation
│   ├── explainer.py     # Plot explanation
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
├── setup.py             # Package configuration
└── README.md            # Project overview
```

Ready to contribute? Start by checking our [GitHub Issues](https://github.com/PlotSenseAI/PlotSense/issues) for good first issues!
