import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from prophet import Prophet
from feature_engine.outliers import OutlierTrimmer
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from yellowbrick.classifier import ClassificationReport
import shap
import optuna
from tqdm import tqdm
import weasyprint
import pdfkit
from jinja2 import Template
import streamlit as st
import warnings
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

# LLM and NLP imports
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text.splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maintenance_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class MaintenanceAnalyzer:
    """Advanced maintenance analysis system with comprehensive analytics capabilities."""
    
    def __init__(
        self,
        data_path: str,
        output_folder: str = "output",
        config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the maintenance analyzer with advanced configuration options.
        
        Args:
            data_path: Path to the input data file
            output_folder: Path to store analysis outputs
            config: Configuration dictionary for customizing analysis parameters
            llm_config: Configuration dictionary for LLM and NLP capabilities
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._initialize_config(config)
        self.llm_config = self._initialize_llm_config(llm_config)
        self.output_folder = Path(output_folder)
        self.figures_folder = self.output_folder / "figures"
        self.models_folder = self.output_folder / "models"
        
        # Create necessary folders
        for folder in [self.output_folder, self.figures_folder, self.models_folder]:
            folder.mkdir(parents=True, exist_ok=True)
            
        # Initialize LLM components
        self._setup_llm_pipeline()
        
        # Load and initialize data
        self.df = self._load_data(data_path)
        self.feature_importances = {}
        self.model_metrics = {}
        self.anomalies = {}
        self.forecasts = {}
        
    def _initialize_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize configuration with default values and user overrides."""
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'n_trials': 100,
            'cv_folds': 5,
            'anomaly_contamination': 0.1,
            'forecast_periods': 30,
            'feature_importance_threshold': 0.01,
            'target_column': None,
            'date_column': None,
            'categorical_threshold': 10,
            'numerical_features': [],
            'categorical_features': [],
            'text_features': [],
        }
        
        if config:
            default_config.update(config)
        return default_config

    def _initialize_llm_config(self, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize LLM configuration with defaults."""
        default_config = {
            'model_name': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_tokens': 500,
            'local_llm': True,  # Use local models by default
            'api_key': None  # OpenAI API key if using their services
        }
        
        if llm_config:
            default_config.update(llm_config)
        return default_config

    def _setup_llm_pipeline(self):
        """Set up LLM pipeline with fallback options."""
        try:
            if self.llm_config['local_llm']:
                # Use local models
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                self.text_classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased"
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.llm_config['embedding_model']
                )
            else:
                # Use OpenAI
                if not self.llm_config['api_key']:
                    raise ValueError("OpenAI API key not provided")
                    
                os.environ['OPENAI_API_KEY'] = self.llm_config['api_key']
                self.llm = OpenAI(
                    temperature=self.llm_config['temperature'],
                    max_tokens=self.llm_config['max_tokens']
                )
                
            # Initialize text splitter for long documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.llm_config['chunk_size'],
                chunk_overlap=self.llm_config['chunk_overlap']
            )
            
        except Exception as e:
            self.logger.warning(f"Error setting up LLM pipeline: {str(e)}")
            self.logger.info("Falling back to basic NLP processing")
            self.llm_enabled = False

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and perform initial data validation."""
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def infer_dataset_properties(self) -> Dict[str, Any]:
        """Automatically infer dataset properties and suggest analysis approaches."""
        properties = {
            'column_types': {},
            'suggested_analyses': [],
            'data_quality': {},
            'potential_targets': []
        }

        try:
            # Infer column types
            for col in self.df.columns:
                dtype = self.df[col].dtype
                nunique = self.df[col].nunique()
                null_pct = self.df[col].isnull().mean()
                
                if pd.api.types.is_numeric_dtype(dtype):
                    if nunique <= self.config['categorical_threshold']:
                        col_type = 'categorical_numeric'
                    else:
                        col_type = 'continuous'
                        if nunique / len(self.df) > 0.9:
                            properties['potential_targets'].append(col)
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = 'datetime'
                elif pd.api.types.is_string_dtype(dtype):
                    if nunique <= self.config['categorical_threshold']:
                        col_type = 'categorical'
                    else:
                        col_type = 'text'
                else:
                    col_type = 'other'
                
                properties['column_types'][col] = {
                    'type': col_type,
                    'unique_values': nunique,
                    'null_percentage': null_pct
                }

            # Suggest analyses based on column types
            if any(info['type'] == 'continuous' for info in properties['column_types'].values()):
                properties['suggested_analyses'].extend([
                    'correlation_analysis',
                    'anomaly_detection',
                    'predictive_modeling'
                ])
            
            if any(info['type'] == 'datetime' for info in properties['column_types'].values()):
                properties['suggested_analyses'].append('time_series_analysis')
            
            if any(info['type'] == 'text' for info in properties['column_types'].values()):
                properties['suggested_analyses'].append('text_analysis')

            # Generate insights using LLM
            if hasattr(self, 'llm'):
                insight_prompt = PromptTemplate(
                    input_variables=["properties"],
                    template="""
                    Based on these dataset properties:
                    {properties}
                    
                    Please provide:
                    1. Recommended analysis approaches
                    2. Potential data quality issues to address
                    3. Key metrics to focus on
                    """
                )
                chain = LLMChain(llm=self.llm, prompt=insight_prompt)
                insights = chain.run(properties=str(properties))
                properties['llm_insights'] = insights.split("\n")

        except Exception as e:
            self.logger.warning(f"Error inferring dataset properties: {str(e)}")

        return properties

    def run_flexible_analysis(self) -> Dict[str, Any]:
        """Run a flexible analysis pipeline based on inferred dataset properties."""
        results = {}
        
        try:
            # Infer dataset properties
            properties = self.infer_dataset_properties()
            results['dataset_properties'] = properties
            
            # Prepare data based on inferred types
            self._prepare_data_by_type(properties['column_types'])
            
            # Run relevant analyses based on suggestions
            analyses = properties['suggested_analyses']
            
            if 'correlation_analysis' in analyses:
                results['correlations'] = self._analyze_correlations()
            
            if 'anomaly_detection' in analyses:
                results['anomalies'] = self._detect_anomalies()
            
            if 'time_series_analysis' in analyses:
                results['time_series'] = self._analyze_time_patterns()
            
            if 'text_analysis' in analyses:
                results['text_insights'] = self._analyze_text_fields()
            
            if 'predictive_modeling' in analyses:
                results['predictions'] = self._build_predictive_models()
            
            # Generate comprehensive insights
            if hasattr(self, 'llm'):
                results['insights'] = self.generate_business_analysis()

        except Exception as e:
            self.logger.error(f"Error in flexible analysis: {str(e)}")
            
        return results

    def _prepare_data_by_type(self, column_types: Dict[str, Dict[str, Any]]) -> None:
        """Prepare data based on inferred column types."""
        for col, info in column_types.items():
            if info['null_percentage'] > 0:
                if info['type'] in ['continuous', 'categorical_numeric']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif info['type'] in ['categorical', 'text']:
                    self.df[col].fillna('MISSING', inplace=True)
            
            if info['type'] == 'categorical':
                self.df[col] = self.df[col].astype('category')

    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numerical columns."""
        numerical_cols = [
            col for col, info in self.inferred_properties['column_types'].items()
            if info['type'] in ['continuous', 'categorical_numeric']
        ]
        
        if not numerical_cols:
            return {}
            
        corr_matrix = self.df[numerical_cols].corr()
        
        # Find strong correlations
        strong_corr = (
            corr_matrix.unstack()
            .drop_duplicates()
            .sort_values(ascending=False)
            .to_dict()
        )
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': {
                k: v for k, v in strong_corr.items() 
                if abs(v) > 0.7 and k[0] != k[1]
            }
        }

    def _analyze_time_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        datetime_cols = [
            col for col, info in self.inferred_properties['column_types'].items()
            if info['type'] == 'datetime'
        ]
        
        if not datetime_cols:
            return {}
            
        patterns = {}
        for date_col in datetime_cols:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Analyze patterns for each numeric column
            numeric_cols = [
                col for col, info in self.inferred_properties['column_types'].items()
                if info['type'] in ['continuous', 'categorical_numeric']
            ]
            
            for num_col in numeric_cols:
                # Time-based aggregations
                patterns[f"{date_col}_{num_col}"] = {
                    'daily_avg': self.df.groupby(self.df[date_col].dt.date)[num_col].mean().to_dict(),
                    'monthly_trend': self.df.groupby(self.df[date_col].dt.to_period('M'))[num_col].agg(['mean', 'std']).to_dict(),
                    'day_of_week_pattern': self.df.groupby(self.df[date_col].dt.day_name())[num_col].mean().to_dict()
                }
                
        return patterns

    def _analyze_text_fields(self) -> Dict[str, Any]:
        """Analyze text fields using NLP and LLM techniques."""
        text_cols = [
            col for col, info in self.inferred_properties['column_types'].items()
            if info['type'] == 'text'
        ]
        
        if not text_cols:
            return {}
            
        results = {}
        for col in text_cols:
            texts = self.df[col].dropna().astype(str).tolist()
            
            # Basic NLP analysis
            docs = list(nlp.pipe(texts[:1000]))  # Analyze first 1000 for performance
            
            # Extract entities and keywords
            entities = {}
            keywords = []
            for doc in docs:
                # Collect named entities
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    entities[ent.label_].append(ent.text)
                
                # Extract important keywords
                keywords.extend([token.text for token in doc if not token.is_stop and token.is_alpha])
            
            # Get sentiment if available
            sentiments = None
            if hasattr(self, 'sentiment_analyzer'):
                sentiments = self.sentiment_analyzer(texts[:100])
            
            # Generate insights using LLM
            insights = []
            if hasattr(self, 'llm'):
                text_sample = "\n".join(texts[:50])  # Analyze a sample
                prompt = PromptTemplate(
                    input_variables=["text", "column"],
                    template="""
                    Analyze this text data from column '{column}' and provide business insights:
                    {text}
                    
                    Please identify:
                    1. Key themes and patterns
                    2. Notable trends or issues
                    3. Business recommendations
                    """
                )
                chain = LLMChain(llm=self.llm, prompt=prompt)
                insights = chain.run(text=text_sample, column=col).split("\n")
            
            results[col] = {
                'entities': entities,
                'top_keywords': pd.Series(keywords).value_counts().head(20).to_dict(),
                'sentiment_distribution': pd.DataFrame(sentiments).label.value_counts().to_dict() if sentiments else None,
                'insights': insights
            }
            
        return results

    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in numerical data using multiple methods."""
        numerical_cols = [
            col for col, info in self.inferred_properties['column_types'].items()
            if info['type'] == 'continuous'
        ]
        
        if not numerical_cols:
            return {}
            
        results = {}
        df_numerical = self.df[numerical_cols].copy()
        
        # Statistical approach (Z-score)
        z_scores = np.abs(stats.zscore(df_numerical))
        statistical_anomalies = (z_scores > 3).any(axis=1)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config['anomaly_contamination'],
            random_state=self.config['random_state']
        )
        isolation_forest_anomalies = iso_forest.fit_predict(df_numerical) == -1
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=self.config['anomaly_contamination'])
        lof_anomalies = lof.fit_predict(df_numerical) == -1
        
        # Combine results
        results['anomaly_indices'] = {
            'statistical': np.where(statistical_anomalies)[0].tolist(),
            'isolation_forest': np.where(isolation_forest_anomalies)[0].tolist(),
            'lof': np.where(lof_anomalies)[0].tolist()
        }
        
        # Generate anomaly insights
        for col in numerical_cols:
            col_anomalies = self.df[col][isolation_forest_anomalies]
            results[col] = {
                'anomaly_values': col_anomalies.tolist(),
                'anomaly_stats': col_anomalies.describe().to_dict()
            }
        
        # Generate LLM insights about anomalies
        if hasattr(self, 'llm'):
            anomaly_prompt = PromptTemplate(
                input_variables=["anomalies"],
                template="""
                Based on these anomaly detection results:
                {anomalies}
                
                Please provide:
                1. Key patterns in the anomalies
                2. Potential business implications
                3. Recommended actions
                """
            )
            chain = LLMChain(llm=self.llm, prompt=anomaly_prompt)
            results['insights'] = chain.run(anomalies=str(results)).split("\n")
        
        return results

    def _build_predictive_models(self) -> Dict[str, Any]:
        """Build and evaluate predictive models for potential target variables."""
        if not self.inferred_properties['potential_targets']:
            return {}
            
        results = {}
        for target in self.inferred_properties['potential_targets']:
            try:
                # Prepare features
                feature_cols = [
                    col for col in self.df.columns 
                    if col != target and 
                    self.inferred_properties['column_types'][col]['type'] 
                    in ['continuous', 'categorical_numeric']
                ]
                
                if not feature_cols:
                    continue
                    
                X = self.df[feature_cols]
                y = self.df[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config['test_size'],
                    random_state=self.config['random_state']
                )
                
                # Train multiple models
                models = {
                    'random_forest': RandomForestRegressor(random_state=self.config['random_state']),
                    'lightgbm': lgb.LGBMRegressor(random_state=self.config['random_state']),
                    'xgboost': xgb.XGBRegressor(random_state=self.config['random_state'])
                }
                
                model_results = {}
                for name, model in models.items():
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = {
                        'r2': r2_score(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'cv_scores': cross_val_score(
                            model, X, y, 
                            cv=self.config['cv_folds']
                        ).tolist()
                    }
                    
                    # Get feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(feature_cols, model.feature_importances_))
                        metrics['feature_importance'] = {
                            k: v for k, v in importance.items() 
                            if v > self.config['feature_importance_threshold']
                        }
                    
                    model_results[name] = metrics
                
                results[target] = model_results
                
                # Generate LLM insights about predictions
                if hasattr(self, 'llm'):
                    prediction_prompt = PromptTemplate(
                        input_variables=["target", "results"],
                        template="""
                        Analyzing predictive modeling results for {target}:
                        {results}
                        
                        Please provide:
                        1. Model performance analysis
                        2. Key predictive factors
                        3. Recommendations for improvement
                        4. Business implications
                        """
                    )
                    chain = LLMChain(llm=self.llm, prompt=prediction_prompt)
                    results[f"{target}_insights"] = chain.run(
                        target=target,
                        results=str(model_results)
                    ).split("\n")
                
            except Exception as e:
                self.logger.warning(f"Error building predictive models for {target}: {str(e)}")
                
        return results

    def _generate_visualizations(self, analysis_results: Dict[str, Any]) -> None:
        """Generate visualizations based on analysis results."""
        if not os.path.exists(self.figures_folder):
            os.makedirs(self.figures_folder)
            
        try:
            # Correlation heatmap
            if 'correlations' in analysis_results:
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pd.DataFrame(analysis_results['correlations']['correlation_matrix']),
                    annot=True,
                    cmap='coolwarm'
                )
                plt.title('Feature Correlations')
                plt.tight_layout()
                plt.savefig(self.figures_folder / 'correlation_matrix.png')
                plt.close()
            
            # Time series patterns
            if 'time_series' in analysis_results:
                for key, patterns in analysis_results['time_series'].items():
                    fig = go.Figure()
                    
                    # Plot daily averages
                    dates = list(patterns['daily_avg'].keys())
                    values = list(patterns['daily_avg'].values())
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name='Daily Average'
                    ))
                    
                    fig.update_layout(
                        title=f'Time Series Analysis: {key}',
                        xaxis_title='Date',
                        yaxis_title='Value'
                    )
                    
                    fig.write_html(str(self.figures_folder / f'{key}_time_series.html'))
            
            # Anomaly visualizations
            if 'anomalies' in analysis_results:
                for col in analysis_results['anomalies'].keys():
                    if col != 'insights' and col != 'anomaly_indices':
                        plt.figure(figsize=(10, 6))
                        
                        # Plot normal vs anomaly distributions
                        sns.kdeplot(
                            data=self.df[col],
                            label='Normal'
                        )
                        sns.kdeplot(
                            data=analysis_results['anomalies'][col]['anomaly_values'],
                            label='Anomalies'
                        )
                        
                        plt.title(f'Anomaly Distribution: {col}')
                        plt.legend()
                        plt.savefig(self.figures_folder / f'{col}_anomalies.png')
                        plt.close()
            
            # Model performance visualizations
            if 'predictions' in analysis_results:
                for target, models in analysis_results['predictions'].items():
                    # Compare model performances
                    model_names = list(models.keys())
                    r2_scores = [models[m]['r2'] for m in model_names]
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=model_names, y=r2_scores)
                    plt.title(f'Model Performance Comparison: {target}')
                    plt.ylabel('R² Score')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(self.figures_folder / f'{target}_model_comparison.png')
                    plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {str(e)}")

    def generate_report(self) -> None:
        """Generate a comprehensive HTML report with interactive visualizations."""
        try:
            # Prepare report sections
            report_sections = {
                'overview': self._generate_overview_section(),
                'data_quality': self._generate_data_quality_section(),
                'analysis': self._generate_analysis_section(),
                'insights': self._generate_insights_section(),
                'recommendations': self._generate_recommendations_section()
            }
            
            # Create HTML template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Business Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .section { margin-bottom: 30px; }
                    .visualization { margin: 20px 0; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f5f5f5; }
                    .insight { padding: 10px; background-color: #f9f9f9; margin: 5px 0; }
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Business Analysis Report</h1>
                <div class="section">
                    <h2>Overview</h2>
                    {{ overview | safe }}
                </div>
                
                <div class="section">
                    <h2>Data Quality Assessment</h2>
                    {{ data_quality | safe }}
                </div>
                
                <div class="section">
                    <h2>Analysis Results</h2>
                    {{ analysis | safe }}
                </div>
                
                <div class="section">
                    <h2>Key Insights</h2>
                    {{ insights | safe }}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {{ recommendations | safe }}
                </div>
            </body>
            </html>
            """
            
            # Generate HTML report
            template = Template(template_str)
            html_content = template.render(**report_sections)
            
            # Save report
            report_path = self.output_folder / 'analysis_report.html'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Convert to PDF if wkhtmltopdf is installed
            try:
                pdf_path = self.output_folder / 'analysis_report.pdf'
                pdfkit.from_file(str(report_path), str(pdf_path))
            except Exception as e:
                self.logger.warning(f"Could not generate PDF: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise

    def _generate_overview_section(self) -> str:
        """Generate HTML for the overview section."""
        overview = f"""
        <p>Dataset Size: {self.df.shape[0]} rows, {self.df.shape[1]} columns</p>
        <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h3>Column Types:</h3>
        <table>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Missing Values (%)</th>
                <th>Unique Values</th>
            </tr>
        """
        
        for col, info in self.inferred_properties['column_types'].items():
            overview += f"""
            <tr>
                <td>{col}</td>
                <td>{info['type']}</td>
                <td>{info['null_percentage']:.2%}</td>
                <td>{info['unique_values']}</td>
            </tr>
            """
        
        overview += "</table>"
        return overview

    def _generate_data_quality_section(self) -> str:
        """Generate HTML for the data quality section."""
        quality_html = "<h3>Data Quality Issues:</h3><ul>"
        
        # Check for missing values
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            quality_html += "<li>Missing values found in columns: " + ", ".join(missing_cols) + "</li>"
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            quality_html += f"<li>Found {duplicates} duplicate rows</li>"
        
        quality_html += "</ul>"
        
        # Add data quality score if available
        if hasattr(self, 'data_quality_score'):
            quality_html += f"<p>Overall Data Quality Score: {self.data_quality_score:.2%}</p>"
        
        return quality_html

    def _generate_analysis_section(self) -> str:
        """Generate HTML for the analysis results section."""
        analysis_html = ""
        
        # Add correlation analysis if available
        if hasattr(self, 'analysis_results') and 'correlations' in self.analysis_results:
            analysis_html += "<h3>Strong Correlations:</h3><ul>"
            for (col1, col2), corr in self.analysis_results['correlations']['strong_correlations'].items():
                analysis_html += f"<li>{col1} - {col2}: {corr:.3f}</li>"
            analysis_html += "</ul>"
        
        # Add time series patterns if available
        if hasattr(self, 'analysis_results') and 'time_series' in self.analysis_results:
            analysis_html += "<h3>Time Series Patterns:</h3>"
            for key, patterns in self.analysis_results['time_series'].items():
                analysis_html += f"""
                <div class="visualization">
                    <iframe src="{self.figures_folder}/{key}_time_series.html" 
                            width="100%" height="400px" frameborder="0"></iframe>
                </div>
                """
        
        # Add model performance if available
        if hasattr(self, 'analysis_results') and 'predictions' in self.analysis_results:
            analysis_html += "<h3>Model Performance:</h3>"
            for target, models in self.analysis_results['predictions'].items():
                analysis_html += f"<h4>Target: {target}</h4><table>"
                analysis_html += "<tr><th>Model</th><th>R² Score</th><th>MSE</th></tr>"
                
                for model_name, metrics in models.items():
                    analysis_html += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{metrics['r2']:.3f}</td>
                        <td>{metrics['mse']:.3f}</td>
                    </tr>
                    """
                analysis_html += "</table>"
        
        return analysis_html

    def _generate_insights_section(self) -> str:
        """Generate HTML for the insights section."""
        insights_html = ""
        
        if hasattr(self, 'analysis_results') and 'insights' in self.analysis_results:
            insights = self.analysis_results['insights']
            insights_html += "<div class='insights'>"
            
            # Add LLM-generated insights
            if isinstance(insights, list):
                for insight in insights:
                    insights_html += f"<div class='insight'>{insight}</div>"
            elif isinstance(insights, dict):
                for category, category_insights in insights.items():
                    insights_html += f"<h3>{category}</h3>"
                    for insight in category_insights:
                        insights_html += f"<div class='insight'>{insight}</div>"
            
            insights_html += "</div>"
        
        return insights_html

    def _generate_recommendations_section(self) -> str:
        """Generate HTML for the recommendations section."""
        recommendations_html = "<ul>"
        
        if hasattr(self, 'analysis_results'):
            # Add data quality recommendations
            if 'data_quality' in self.analysis_results:
                recommendations_html += "<h3>Data Quality Improvements:</h3><ul>"
                for rec in self.analysis_results['data_quality'].get('recommendations', []):
                    recommendations_html += f"<li>{rec}</li>"
                recommendations_html += "</ul>"
            
            # Add business recommendations
            if 'insights' in self.analysis_results:
                recommendations_html += "<h3>Business Recommendations:</h3><ul>"
                for rec in self.analysis_results['insights'].get('recommendations', []):
                    recommendations_html += f"<li>{rec}</li>"
                recommendations_html += "</ul>"
        
        recommendations_html += "</ul>"
        return recommendations_html

    def analyze_dataset(self, data_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze any dataset with automatic inference and flexible processing."""
        try:
            # Update configuration if provided
            if config:
                self.config.update(config)
            
            # Load new dataset
            self.df = self._load_data(data_path)
            
            # Infer properties and run flexible analysis
            self.inferred_properties = self.infer_dataset_properties()
            results = self.run_flexible_analysis()
            
            # Generate visualizations based on analysis results
            self._generate_visualizations(results)
            
            # Create report
            self.generate_report()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset: {str(e)}")
            raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up configuration
    config = {
        'target_column': 'maintenance_cost',
        'date_column': 'maintenance_date',
        'random_state': 42,
        'test_size': 0.2
    }
    
    llm_config = {
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'local_llm': True
    }
    
    # Initialize and run analyzer
    analyzer = MaintenanceAnalyzer(
        data_path="sample.csv",
        output_folder="advanced_analysis_results",
        config=config,
        llm_config=llm_config
    )
    
    analyzer.run_analysis()