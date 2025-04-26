import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import datetime

warnings.filterwarnings('ignore')

class DynamicMaintenanceAnalyzer:
    def __init__(self, data_path, output_folder="output"):
        self.df = pd.read_csv(data_path)
        self.insights = []
        self.recommendations = []
        self.summary = {}
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base")
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_data(self):
        print("Step 1: Preprocessing Data...")
        # Handle missing values dynamically
        self.df.fillna(0, inplace=True)

        # Dynamically detect column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = [col for col in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[col])]

        # Convert suspected date-like strings to datetime
        for col in tqdm(self.df.columns, desc="Converting Date Columns"):
            if self.df[col].dtype == 'object' and any(keyword in col.lower() for keyword in ['date', 'time']):
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    self.date_cols.append(col)
                except Exception:
                    pass

    def generate_data_summary(self):
        print("\nStep 2: Generating Data Summary...")
        # Create a summary of the dataset dynamically
        self.summary = {
            "rows": len(self.df),
            "columns": self.df.columns.tolist(),
            "numerical_stats": {col: self.df[col].describe().to_dict() for col in tqdm(self.numerical_cols, desc="Summarizing Numerical Columns")},
            "categorical_stats": {col: self.df[col].value_counts().to_dict() for col in tqdm(self.categorical_cols, desc="Summarizing Categorical Columns")},
            "missing_values": {col: self.df[col].isnull().sum() for col in tqdm(self.df.columns, desc="Checking Missing Values")}
        }

        # Generate a human-readable summary for the report
        summary_html = f"""
        <h3>Dataset Overview</h3>
        <ul>
            <li><strong>Rows:</strong> {self.summary['rows']}</li>
            <li><strong>Columns:</strong> {len(self.summary['columns'])}</li>
        </ul>

        <h3>Columns</h3>
        <ul>
            {''.join([f'<li>{col}</li>' for col in self.summary['columns']])}
        </ul>

        <h3>Numerical Statistics</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Std</th>
                </tr>
            </thead>
            <tbody>
                {''.join([
                    f'<tr>'
                    f'<td>{col}</td>'
                    f'<td>{stats.get("mean", "N/A"):.2f}</td>'
                    f'<td>{stats.get("min", "N/A")}</td>'
                    f'<td>{stats.get("max", "N/A")}</td>'
                    f'<td>{stats.get("std", "N/A"):.2f}</td>'
                    f'</tr>'
                    for col, stats in self.summary["numerical_stats"].items()
                ])}
            </tbody>
        </table>

        <h3>Categorical Statistics</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Top Categories</th>
                    <th>Counts</th>
                </tr>
            </thead>
            <tbody>
                {''.join([
                    f'<tr>'
                    f'<td>{col}</td>'
                    f'<td>{", ".join([str(cat) for cat in list(stats.keys())[:5]])}</td>'
                    f'<td>{", ".join([str(count) for count in list(stats.values())[:5]])}</td>'
                    f'</tr>'
                    for col, stats in self.summary["categorical_stats"].items()
                ])}
            </tbody>
        </table>

        <h3>Missing Values</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Missing Values</th>
                </tr>
            </thead>
            <tbody>
                {''.join([
                    f'<tr>'
                    f'<td>{col}</td>'
                    f'<td>{count}</td>'
                    f'</tr>'
                    for col, count in self.summary["missing_values"].items()
                ])}
            </tbody>
        </table>
        """

        # Store the formatted summary in the class for later use in the report
        self.formatted_summary = summary_html

    def dynamic_eda(self):
        print("\nStep 3: Performing Exploratory Data Analysis (EDA)...")
        # Generate visualizations dynamically based on column types
        plt.figure(figsize=(15, 10))

        # Plot numerical distributions
        for i, col in enumerate(tqdm(self.numerical_cols[:min(4, len(self.numerical_cols))], desc="Plotting Numerical Distributions"), 1):
            plt.subplot(2, 2, i)
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'numerical_distributions.png'))
        plt.close()

        # Plot categorical counts
        if self.categorical_cols:
            plt.figure(figsize=(15, 6))
            for i, col in enumerate(tqdm(self.categorical_cols[:min(3, len(self.categorical_cols))], desc="Plotting Categorical Counts"), 1):
                plt.subplot(1, 3, i)
                top_categories = self.df[col].value_counts().iloc[:10]
                sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
                plt.title(f'Top Categories in {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, 'categorical_counts.png'))
            plt.close()

        # Time series analysis for date columns
        if self.date_cols:
            plt.figure(figsize=(15, 6))
            for i, col in enumerate(tqdm(self.date_cols[:min(3, len(self.date_cols))], desc="Plotting Time Series"), 1):
                plt.subplot(1, 3, i)
                try:
                    # Ensure the datetime column is preserved for resampling
                    temp_df = self.df.copy()
                    temp_df.set_index(col, inplace=True)

                    # Select only numeric columns for resampling
                    numeric_df = temp_df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        resampled = numeric_df.resample('M').mean()
                        if not resampled.empty:
                            resampled.plot(ax=plt.gca())
                            plt.title(f'Monthly Trends for {col}')
                        else:
                            plt.title(f"No resampled data available for {col}")
                    else:
                        plt.title(f"No numeric data available for {col}")
                except Exception as e:
                    plt.title(f"Error plotting {col}: {str(e)}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, 'time_series.png'))
            plt.close()

    def llm_analysis(self):
        print("\nStep 4: Generating Insights Using LLM...")
        # Truncate or summarize the dataset summary to fit within the token limit
        max_prompt_length = 500  # Maximum characters allowed for the prompt

        truncated_summary = {
            "rows": self.summary['rows'],
            "columns": self.summary['columns'][:10],  # Show only the first 10 columns
            "numerical_stats": {col: stats for col, stats in list(self.summary['numerical_stats'].items())[:3]},  # Show stats for only 3 numeric columns
            "categorical_stats": {col: stats for col, stats in list(self.summary['categorical_stats'].items())[:3]},  # Show stats for only 3 categorical columns
            "missing_values": {col: val for col, val in list(self.summary['missing_values'].items())[:5]}  # Show missing values for only 5 columns
        }

        prompt = f"""
        Analyze the following dataset summary and provide business insights and recommendations:
        - Rows: {truncated_summary['rows']}
        - Columns: {truncated_summary['columns']}
        - Key Numerical Stats: {truncated_summary['numerical_stats']}
        - Key Categorical Stats: {truncated_summary['categorical_stats']}
        - Missing Values: {truncated_summary['missing_values']}
        """

        # Ensure the prompt does not exceed the maximum length
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length]

        try:
            llm_response = self.llm(prompt, max_length=500)[0]['generated_text']
            self.insights = [insight.strip() for insight in llm_response.split('\n') if insight]
        except Exception as e:
            print(f"Error during LLM analysis: {str(e)}")
            self.insights.append("Unable to generate insights due to input size or model limitations.")

    def ml_analysis(self):
        print("\nStep 5: Applying Machine Learning Models...")
        # Optional ML models for predictive analysis
        if len(self.numerical_cols) >= 2:
            # Example: Predictive maintenance cost (if applicable)
            target_col = None
            for col in tqdm(self.numerical_cols, desc="Detecting Target Column"):
                if any(keyword in col.lower() for keyword in ['cost', 'expense', 'price']):
                    target_col = col
                    break

            if target_col:
                X = self.df[self.numerical_cols].drop(target_col, axis=1)
                y = self.df[target_col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression().fit(X_train, y_train)
                score = model.score(X_test, y_test)
                self.recommendations.append(f"Predicted {target_col} with R2 score: {score:.2f}")

        # Clustering example (if applicable)
        if len(self.numerical_cols) >= 2:
            kmeans = KMeans(n_clusters=min(3, len(self.df))).fit(self.df[self.numerical_cols])
            self.df['cluster'] = kmeans.labels_
            self.recommendations.append("Clustered equipment for optimized scheduling.")

    def generate_report(self):
        print("\nStep 6: Generating Final Report...")
        # CSS for styling the report
        css_style = """
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                background-color: #f9f9f9;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 28px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            table th, table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            table th {
                background-color: #f2f2f2;
            }
            ul {
                list-style-type: disc;
                margin-left: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                font-size: 14px;
                color: #777;
            }
        </style>
        """

        # Define the Jinja2 template as a standalone string
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Maintenance Analysis Report</title>
            {{ css_style }}
        </head>
        <body>
            <div class="container">
                <h1>Maintenance Analysis Report</h1>

                <h2>Data Overview</h2>
                {{ summary|safe }}

                <h2>Insights</h2>
                <ul>{{ insights|safe }}</ul>

                <h2>Recommendations</h2>
                <ul>{{ recommendations|safe }}</ul>

                <h2>Visualizations</h2>
                {% if numerical_cols %}
                    <img src='numerical_distributions.png' alt='Numerical Distributions'>
                {% endif %}
                {% if categorical_cols %}
                    <img src='categorical_counts.png' alt='Categorical Counts'>
                {% endif %}
                {% if date_cols %}
                    <img src='time_series.png' alt='Time Series Trends'>
                {% endif %}
            </div>
            <div class="footer">
                Generated by Maintenance Analyzer | {{ timestamp }}
            </div>
        </body>
        </html>
        """

        # Render the report using Jinja2
        from jinja2 import Template
        template = Template(html_template)

        # Pass dynamic content to the template
        html_report = template.render(
            css_style=css_style,
            summary=self.formatted_summary,
            insights="<li>" + "</li><li>".join(self.insights) + "</li>",
            recommendations="<li>" + "</li><li>".join(self.recommendations) + "</li>",
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            date_cols=self.date_cols,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Save the HTML report in the output folder
        with open(os.path.join(self.output_folder, "report.html"), "w", encoding="utf-8") as file:
            file.write(html_report)

    def analyze(self):
        print("Starting Analysis Pipeline...\n")
        self.preprocess_data()
        self.generate_data_summary()
        self.dynamic_eda()
        self.llm_analysis()
        self.ml_analysis()
        self.generate_report()
        print(f"\nAnalysis Complete! Check the generated outputs in the '{self.output_folder}' folder.")

# Usage
if __name__ == "__main__":
    analyzer = DynamicMaintenanceAnalyzer("sample2.csv", output_folder="output_results_project1_dataset2")
    analyzer.analyze()