import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
from datetime import datetime
import os
import shutil
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import torch
warnings.filterwarnings('ignore')

# Download required NLTK data with comprehensive error handling
def download_nltk_data():
    """Download all required NLTK data with comprehensive error handling"""
    required_packages = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                print(f"Downloading required NLTK package: {package}")
                nltk.download(package, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download {package}. Using fallback tokenization. Error: {str(e)}")

# Download NLTK data at startup
download_nltk_data()

class MaintenanceAnalyzer:
    def __init__(self, file_path):
        print("Loading data...")
        self.df = pd.read_csv(file_path)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_results')
        
        # Auto-detect and map columns
        self._detect_and_map_columns()
        
        # Create output directory
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                print(f"Warning: Could not clean old files. {str(e)}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self._preprocess_data()
        
    def _detect_and_map_columns(self):
        """Automatically detect and map columns based on content and patterns"""
        print("Detecting column types...")
        
        # Initialize column categories
        self.column_types = {
            'cost': None,
            'equipment': None,
            'task': None,
            'start_date': None,
            'end_date': None,
            'duration': None,
            'description': None
        }
        
        # Function to check if a column contains dates
        def is_date_column(series):
            try:
                pd.to_datetime(series.dropna().head())
                return True
            except:
                return False
        
        # Function to check if a column contains monetary values
        def is_cost_column(series):
            if series.dtype in ['float64', 'int64']:
                return True
            if series.dtype == 'object':
                sample = series.dropna().head().astype(str)
                return any('$' in str(x) or ',' in str(x) for x in sample)
            return False
        
        # Analyze each column
        for col in self.df.columns:
            col_lower = col.lower()
            sample_values = self.df[col].dropna().astype(str).head()
            
            # Detect cost columns
            if (is_cost_column(self.df[col]) and 
                any(term in col_lower for term in ['cost', 'price', 'amount', 'charge'])):
                self.column_types['cost'] = col
            
            # Detect equipment columns
            elif any(term in col_lower for term in ['equipment', 'machine', 'asset', 'device']):
                self.column_types['equipment'] = col
            
            # Detect task ID columns
            elif any(term in col_lower for term in ['task', 'work order', 'maintenance']):
                self.column_types['task'] = col
            
            # Detect date columns
            elif is_date_column(self.df[col]):
                if 'start' in col_lower:
                    self.column_types['start_date'] = col
                elif 'end' in col_lower:
                    self.column_types['end_date'] = col
            
            # Detect duration columns
            elif any(term in col_lower for term in ['duration', 'hours', 'time']):
                self.column_types['duration'] = col
            
            # Detect description columns
            elif any(term in col_lower for term in ['description', 'desc', 'details', 'notes']):
                self.column_types['description'] = col
        
        # Create standard column names
        self.standard_columns = {
            'cost': 'Maintenance_Cost',
            'equipment': 'Equipment',
            'task': 'Task_ID',
            'start_date': 'Start_Date',
            'end_date': 'End_Date',
            'duration': 'Duration',
            'description': 'Description'
        }
        
        # Rename detected columns
        rename_map = {
            self.column_types[k]: v 
            for k, v in self.standard_columns.items() 
            if self.column_types[k] is not None
        }
        self.df = self.df.rename(columns=rename_map)
        
        # Ensure we have minimum required columns
        if not any(col in self.df.columns for col in [self.standard_columns['equipment'], self.standard_columns['cost']]):
            raise ValueError("Dataset must contain at least equipment and cost information")
    
    def _preprocess_data(self):
        """Preprocess the data based on detected columns"""
        print("Preprocessing data...")
        
        # Handle dates
        date_cols = [col for col in [self.standard_columns['start_date'], self.standard_columns['end_date']] 
                    if col in self.df.columns]
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Handle duration
        if self.standard_columns['duration'] not in self.df.columns:
            if all(col in self.df.columns for col in [self.standard_columns['start_date'], self.standard_columns['end_date']]):
                self.df[self.standard_columns['duration']] = (
                    self.df[self.standard_columns['end_date']] - 
                    self.df[self.standard_columns['start_date']]
                ).dt.total_seconds() / 3600  # Convert to hours
        
        # Handle costs
        if self.standard_columns['cost'] in self.df.columns:
            if self.df[self.standard_columns['cost']].dtype == 'object':
                self.df[self.standard_columns['cost']] = self.df[self.standard_columns['cost']].apply(
                    lambda x: float(str(x).replace('$', '').replace(',', '')) if pd.notnull(x) else np.nan
                )
    
    def generate_basic_stats(self):
        """Generate statistics based on available columns"""
        print("Generating basic statistics...")
        stats = {'total_records': len(self.df)}
        
        if self.standard_columns['equipment'] in self.df.columns:
            stats['unique_equipment'] = self.df[self.standard_columns['equipment']].nunique()
        
        if self.standard_columns['cost'] in self.df.columns:
            stats.update({
                'total_cost': self.df[self.standard_columns['cost']].sum(),
                'avg_cost': self.df[self.standard_columns['cost']].mean(),
                'min_cost': self.df[self.standard_columns['cost']].min(),
                'max_cost': self.df[self.standard_columns['cost']].max()
            })
        
        if self.standard_columns['duration'] in self.df.columns:
            stats['avg_duration'] = self.df[self.standard_columns['duration']].mean()
        
        if all(col in self.df.columns for col in [self.standard_columns['start_date'], self.standard_columns['end_date']]):
            stats['date_range'] = (
                self.df[self.standard_columns['end_date']].max() - 
                self.df[self.standard_columns['start_date']].min()
            ).days
        
        return stats
    
    def analyze_by_category(self, category_col):
        """Analyze data by any categorical column"""
        if category_col not in self.df.columns:
            return None
            
        analysis = self.df.groupby(category_col).agg({
            col: ['count', 'sum', 'mean'] if col == self.standard_columns['cost']
            else 'count' for col in self.df.columns 
            if col == self.standard_columns['cost'] or col == self.standard_columns['duration']
        }).round(2)
        
        if len(analysis.columns) > 0:
            # Flatten column names
            analysis.columns = [f"{col[1]}_{col[0]}" for col in analysis.columns]
            return analysis.sort_values(
                f"sum_{self.standard_columns['cost']}" if f"sum_{self.standard_columns['cost']}" in analysis.columns
                else analysis.columns[0], 
                ascending=False
            )
        return None
    
    def create_visualizations(self):
        """Create visualizations based on available data"""
        print("Creating visualizations...")
        
        # 1. Time series or distribution plot
        plt.figure(figsize=(12, 6))
        if self.standard_columns['start_date'] in self.df.columns:
            # Time series plot
            costs_over_time = self.df.groupby(
                pd.Grouper(key=self.standard_columns['start_date'], freq='M')
            )[self.standard_columns['cost']].sum()
            plt.plot(costs_over_time.index, costs_over_time.values)
            plt.title('Maintenance Costs Over Time')
            plt.xlabel('Date')
        else:
            # Cost distribution
            plt.hist(self.df[self.standard_columns['cost']], bins=30)
            plt.title('Distribution of Maintenance Costs')
            plt.xlabel('Cost')
        
        plt.ylabel('Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'cost_analysis.png'))
        plt.close()

        # 2. Category comparison (if equipment column exists)
        if self.standard_columns['equipment'] in self.df.columns:
            plt.figure(figsize=(12, 6))
            category_costs = self.df.groupby(self.standard_columns['equipment'])[self.standard_columns['cost']].sum()
            category_costs.nlargest(10).plot(kind='bar')
            plt.title('Top 10 Categories by Cost')
            plt.xlabel(self.standard_columns['equipment'])
            plt.ylabel('Total Cost')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'category_costs.png'))
            plt.close()

        # 3. Additional visualizations based on available data
        if self.standard_columns['description'] in self.df.columns:
            plt.figure(figsize=(10, 10))
            task_counts = self.df[self.standard_columns['description']].value_counts()
            plt.pie(task_counts.head().values, labels=task_counts.head().index, autopct='%1.1f%%')
            plt.title('Top 5 Maintenance Types')
            plt.axis('equal')
            plt.savefig(os.path.join(self.images_dir, 'maintenance_types.png'))
            plt.close()
    
    def _setup_llm(self):
        """Initialize the LLM model for text analysis"""
        print("Setting up LLM model...")
        model_name = "distilbert-base-uncased"  # Free, lightweight model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=model_name)
        
    def analyze_maintenance_patterns(self):
        """Analyze maintenance descriptions using LLM to identify patterns and insights"""
        if self.standard_columns['description'] not in self.df.columns:
            return None
            
        descriptions = self.df[self.standard_columns['description']].dropna().tolist()
        
        # Text preprocessing with fallback
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()  # Fallback to empty set if stopwords unavailable
            
        processed_texts = []
        for desc in descriptions:
            try:
                tokens = word_tokenize(str(desc).lower())
            except:
                # Fallback to simple splitting if NLTK tokenization fails
                tokens = str(desc).lower().split()
            
            tokens = [t for t in tokens if t not in stop_words and t.isalnum()]
            processed_texts.append(' '.join(tokens))
            
        # Extract key topics using TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Identify common maintenance patterns
        feature_names = vectorizer.get_feature_names_out()  # Updated method name
        word_freq = pd.DataFrame(
            tfidf_matrix.sum(axis=0).T,
            index=feature_names,
            columns=['frequency']
        ).sort_values('frequency', ascending=False)
        
        # Get sentiment analysis for maintenance descriptions
        sentiments = self.nlp(descriptions[:100])  # Analyze first 100 descriptions
        sentiment_stats = pd.DataFrame(sentiments).label.value_counts()
        
        return {
            'common_terms': word_freq.head(10).to_dict()['frequency'],
            'sentiment_analysis': sentiment_stats.to_dict(),
            'sample_insights': self._generate_insights(descriptions[:5])
        }
        
    def _generate_insights(self, sample_descriptions):
        """Generate insights from maintenance descriptions using LLM"""
        insights = []
        for desc in sample_descriptions:
            result = self.nlp(desc)[0]
            insights.append({
                'description': desc,
                'classification': result['label'],
                'confidence': f"{result['score']:.2%}"
            })
        return insights

    def generate_business_recommendations(self):
        """Generate business-wise recommendations based on maintenance data analysis"""
        recommendations = []
        
        # Analyze cost patterns if both date and cost columns are available
        if (self.standard_columns['cost'] in self.df.columns and 
            self.standard_columns['start_date'] in self.df.columns):
            try:
                monthly_costs = self.df.groupby(
                    pd.Grouper(key=self.standard_columns['start_date'], freq='M')
                )[self.standard_columns['cost']].sum()
                
                cost_trend = monthly_costs.pct_change().mean()
                if cost_trend > 0.05:  # 5% increase
                    recommendations.append({
                        'category': 'Cost Management',
                        'observation': f'Maintenance costs are increasing at {cost_trend:.1%} per month',
                        'recommendation': 'Consider implementing preventive maintenance programs to reduce long-term costs',
                        'priority': 'High' if cost_trend > 0.1 else 'Medium'
                    })
            except Exception as e:
                print(f"Warning: Could not analyze cost trends: {str(e)}")
        
        # Basic cost analysis without time series
        if self.standard_columns['cost'] in self.df.columns:
            total_cost = self.df[self.standard_columns['cost']].sum()
            avg_cost = self.df[self.standard_columns['cost']].mean()
            if avg_cost > total_cost * 0.1:  # If average cost is more than 10% of total
                recommendations.append({
                    'category': 'Cost Optimization',
                    'observation': f'High average maintenance cost (${avg_cost:,.2f})',
                    'recommendation': 'Review maintenance procedures and identify cost reduction opportunities',
                    'priority': 'High'
                })
        
        # Equipment analysis
        if self.standard_columns['equipment'] in self.df.columns:
            equipment_costs = self.df.groupby(self.standard_columns['equipment'])[self.standard_columns['cost']].agg(['sum', 'count'])
            high_frequency = equipment_costs[equipment_costs['count'] > equipment_costs['count'].mean() + equipment_costs['count'].std()]
            
            for equip in high_frequency.index:
                recommendations.append({
                    'category': 'Equipment Performance',
                    'observation': f'{equip} requires frequent maintenance',
                    'recommendation': 'Evaluate replacement or upgrade options for this equipment',
                    'priority': 'High'
                })
        
        # Duration analysis
        if self.standard_columns['duration'] in self.df.columns:
            avg_duration = self.df[self.standard_columns['duration']].mean()
            if avg_duration > 24:  # More than 24 hours
                recommendations.append({
                    'category': 'Operational Efficiency',
                    'observation': f'Average maintenance duration is {avg_duration:.1f} hours',
                    'recommendation': 'Review maintenance procedures and consider training programs to reduce downtime',
                    'priority': 'Medium'
                })
        
        # Use LLM for maintenance description analysis
        if self.standard_columns['description'] in self.df.columns:
            descriptions = self.df[self.standard_columns['description']].dropna().tolist()
            
            try:
                # Prepare a prompt for business analysis
                analysis_prompt = "Analyze these maintenance descriptions and provide business recommendations:\n\n"
                analysis_prompt += "\n".join(descriptions[:5])  # Analyze first 5 descriptions for demonstration
                
                # Use the LLM pipeline for analysis
                analysis_result = self.nlp(analysis_prompt)
                
                # Process LLM insights
                for result in analysis_result:
                    if result['score'] > 0.7:  # High confidence insights
                        recommendations.append({
                            'category': 'Process Improvement',
                            'observation': f'Pattern identified: {result["label"]}',
                            'recommendation': self._generate_recommendation_from_label(result["label"]),
                            'priority': 'Medium'
                        })
            except Exception as e:
                print(f"Warning: LLM analysis encountered an error: {str(e)}")
        
        return recommendations

    def _generate_recommendation_from_label(self, label):
        """Convert LLM classification labels into actionable recommendations"""
        recommendation_map = {
            'POSITIVE': 'Continue current maintenance practices for this category',
            'NEGATIVE': 'Review and revise maintenance procedures for this category',
            'NEUTRAL': 'Monitor performance and collect more data for analysis'
        }
        return recommendation_map.get(label, 'Further analysis needed')

    def create_html_report(self):
        """Generate HTML report with LLM analysis"""
        print("Generating enhanced HTML report with LLM analysis...")
        stats = self.generate_basic_stats()
        category_analysis = self.analyze_by_category(self.standard_columns['equipment'])
        self.create_visualizations()
        
        # Initialize LLM and get maintenance patterns
        self._setup_llm()
        maintenance_patterns = self.analyze_maintenance_patterns()
        
        # Create dynamic stats cards based on available data
        stats_cards = []
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                formatted_value = f"${value:,.2f}" if 'cost' in key else f"{value:,.2f}"
                stats_cards.append(f"""
                    <div class="stat-card">
                        <h3>{key.replace('_', ' ').title()}</h3>
                        <p>{formatted_value}</p>
                    </div>
                """)
        
        # Create HTML with dynamic sections
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Maintenance Analysis Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background-color: #f5f5f5; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stats {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px; 
                }}
                .stat-card {{ 
                    padding: 20px; 
                    background: white; 
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .visualization {{ 
                    margin: 30px 0; 
                    text-align: center;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{ max-width: 100%; border-radius: 4px; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0;
                    background: white;
                }}
                th, td {{ 
                    padding: 12px; 
                    border: 1px solid #ddd; 
                    text-align: left;
                }}
                th {{ background: #f8f9fa; }}
                .insight-card {{
                    background: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .confidence {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Maintenance Analysis Report</h1>
                <p>Generated on {datetime.now().strftime("%B %d, %Y")}</p>
            </div>
            
            <div class="stats">
                {''.join(stats_cards)}
            </div>
        """
        
        # Add visualizations based on available data
        if os.path.exists(os.path.join(self.images_dir, 'cost_analysis.png')):
            html_content += """
            <div class="visualization">
                <h2>Cost Analysis</h2>
                <img src="images/cost_analysis.png" alt="Cost Analysis">
            </div>
            """
            
        if os.path.exists(os.path.join(self.images_dir, 'category_costs.png')):
            html_content += """
            <div class="visualization">
                <h2>Category Costs</h2>
                <img src="images/category_costs.png" alt="Category Costs">
            </div>
            """
            
        if os.path.exists(os.path.join(self.images_dir, 'maintenance_types.png')):
            html_content += """
            <div class="visualization">
                <h2>Maintenance Types Distribution</h2>
                <img src="images/maintenance_types.png" alt="Maintenance Types">
            </div>
            """
        
        # Add category analysis table if available
        if category_analysis is not None:
            html_content += f"""
            <div class="visualization">
                <h2>Category Analysis</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        {''.join(f"<th>{col.replace('_', ' ').title()}</th>" for col in category_analysis.columns)}
                    </tr>
                    {''.join(f"""
                    <tr>
                        <td>{category}</td>
                        {''.join(f"<td>${val:,.2f}" if 'cost' in col else f"<td>{val:,.2f}" 
                                for col, val in row.items())}
                    </tr>
                    """ for category, row in category_analysis.head(10).iterrows())}
                </table>
            </div>
            """
        
        # Add LLM analysis section
        if maintenance_patterns:
            html_content += self._generate_llm_analysis_html(maintenance_patterns)
        
        # Get business recommendations
        business_recommendations = self.generate_business_recommendations()
        
        # Add business recommendations section
        if business_recommendations:
            html_content += """
            <div class="visualization">
                <h2>Business Recommendations</h2>
                <style>
                    .recommendation-card {
                        background: white;
                        padding: 20px;
                        margin: 15px 0;
                        border-radius: 8px;
                        border-left: 4px solid;
                    }
                    .priority-High { border-left-color: #dc3545; }
                    .priority-Medium { border-left-color: #ffc107; }
                    .priority-Low { border-left-color: #28a745; }
                </style>
            """
            
            for rec in business_recommendations:
                html_content += f"""
                <div class="recommendation-card priority-{rec['priority']}">
                    <h3>{rec['category']}</h3>
                    <p><strong>Observation:</strong> {rec['observation']}</p>
                    <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                </div>
                """
                
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        report_path = os.path.join(self.output_dir, 'maintenance_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Enhanced report generated successfully! All outputs are saved in the '{self.output_dir}' folder.")
        
    def _generate_llm_analysis_html(self, patterns):
        """Generate HTML section for LLM analysis results"""
        if not patterns:
            return ""
            
        html = """
        <div class="visualization">
            <h2>LLM-Based Maintenance Analysis</h2>
            
            <h3>Common Maintenance Terms</h3>
            <table>
                <tr><th>Term</th><th>Frequency</th></tr>
        """
        
        for term, freq in patterns['common_terms'].items():
            html += f"<tr><td>{term}</td><td>{freq:.2f}</td></tr>"
            
        html += """
            </table>
            
            <h3>Sentiment Analysis</h3>
            <table>
                <tr><th>Category</th><th>Count</th></tr>
        """
        
        for category, count in patterns['sentiment_analysis'].items():
            html += f"<tr><td>{category}</td><td>{count}</td></tr>"
            
        html += """
            </table>
            
            <h3>Sample Maintenance Insights</h3>
        """
        
        for insight in patterns['sample_insights']:
            html += f"""
            <div class="insight-card">
                <p><strong>Description:</strong> {insight['description']}</p>
                <p><strong>Classification:</strong> {insight['classification']}</p>
                <p class="confidence">Confidence: {insight['confidence']}</p>
            </div>
            """
            
        html += "</div>"
        return html

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze maintenance data and generate reports')
    parser.add_argument('file_path', nargs='?', default='sample.csv', 
                      help='Path to the CSV file containing maintenance data')
    
    args = parser.parse_args()
    print(f"Analyzing file: {args.file_path}")
    analyzer = MaintenanceAnalyzer(args.file_path)
    analyzer.create_html_report()