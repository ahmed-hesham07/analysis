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
warnings.filterwarnings('ignore')

class MaintenanceAnalyzer:
    def __init__(self, file_path):
        print("Loading data...")
        self.df = pd.read_csv(file_path)
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_results_project2_dataset2')
        
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
    
    def create_html_report(self):
        """Generate HTML report based on available data"""
        print("Generating HTML report...")
        stats = self.generate_basic_stats()
        category_analysis = self.analyze_by_category(self.standard_columns['equipment'])
        self.create_visualizations()
        
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
            <title>Maintenance Analysis Report</title>
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
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Maintenance Analysis Report</h1>
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
        
        html_content += """
        </body>
        </html>
        """
        
        report_path = os.path.join(self.output_dir, 'maintenance_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated successfully! All outputs are saved in the '{self.output_dir}' folder.")
        print(f"Open '{report_path}' in your browser to view the report.")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze maintenance data and generate reports')
    parser.add_argument('file_path', nargs='?', default='sample2.csv', 
                      help='Path to the CSV file containing maintenance data')
    
    args = parser.parse_args()
    print(f"Analyzing file: {args.file_path}")
    analyzer = MaintenanceAnalyzer(args.file_path)
    analyzer.create_html_report()