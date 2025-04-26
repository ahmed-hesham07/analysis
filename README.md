# Maintenance Analysis Projects

This repository contains two maintenance analysis projects that process and analyze maintenance data using different approaches.

## Project 1 (project1.py)

An advanced maintenance analysis system with comprehensive analytics capabilities. This project uses machine learning, NLP, and statistical analysis to provide deep insights into maintenance data.

### Outputs (from sample.csv):
- **output_results_project1_dataset1/**
  - `anomalies.png` - Visualization of detected anomalies in the data
  - `box_plots.html` - Interactive box plots for numerical distributions
  - `categorical_counts.png` - Distribution of categorical variables
  - `correlation_matrix.png` - Heatmap of feature correlations
  - `interactive_correlation.html` - Interactive correlation analysis
  - `numerical_distributions.png` - Distribution plots for numerical variables
  - `predictive_analysis.html` - Results from predictive modeling
  - `report.html/pdf` - Comprehensive analysis report
  - `scatter_matrix.html` - Interactive scatter plot matrix
  - `time_series.png` - Time series analysis visualization

### Outputs (from sample2.csv):
- **output_results_project1_dataset2/**
  - `categorical_counts.png` - Distribution of categorical variables
  - `numerical_distributions.png` - Distribution plots for numerical variables
  - `report.html/pdf` - Analysis report for the second dataset

### Key Features:
- Automatic column type detection and mapping
- Advanced anomaly detection using multiple methods
- Time series analysis and pattern detection
- Predictive modeling with multiple algorithms
- Text analysis using NLP and LLM techniques
- Interactive visualizations and comprehensive reporting

## Project 2 (project2.py)

A focused maintenance analysis system that specializes in cost analysis and business recommendations.

### Outputs (from sample.csv):
- **output_results/**
  - `maintenance_report.html/pdf` - Detailed maintenance analysis report
  - **images/**
    - `category_costs.png` - Analysis of costs by category
    - `cost_analysis.png` - Time series and distribution of costs

### Outputs (from sample2.csv):
- **output_results2/**
  - `maintenance_report.html/pdf` - Detailed maintenance analysis report
  - **images/**
    - `category_costs.png` - Analysis of costs by category
    - `cost_analysis.png` - Time series and distribution of costs
    - `maintenance_types.png` - Distribution of maintenance types

### Key Features:
- Automated column detection and mapping
- Cost analysis and visualization
- Business recommendations generation
- LLM-based pattern analysis
- PDF report generation with enhanced formatting

## Requirements
- Python 3.x
- Required packages listed in requirements.txt
- NLTK data packages
- SpaCy model (en_core_web_sm)
- wkhtmltopdf (for PDF generation)

## Usage
```python
# For Project 1
analyzer = MaintenanceAnalyzer(
    data_path="sample.csv",
    output_folder="output_results_project1_dataset1"
)
analyzer.analyze_dataset()

# For Project 2
analyzer = MaintenanceAnalyzer("sample.csv")
analyzer.create_html_report()
```

## Output Structure
Both projects generate comprehensive reports in HTML and PDF formats, along with supporting visualizations. The reports include:
- Executive summary
- Data quality assessment
- Statistical analysis
- Visualizations
- Business recommendations
- LLM-generated insights