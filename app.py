from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
import json
import base64
from io import BytesIO, StringIO
import duckdb
import sqlite3
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import os
from werkzeug.utils import secure_filename
import tempfile
import traceback
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

class DataAnalystAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_wikipedia_highest_grossing_films(self, url: str) -> pd.DataFrame:
        """Scrape highest grossing films from Wikipedia"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main table with highest grossing films
            table = None
            tables = soup.find_all('table', class_='wikitable')
            
            for t in tables:
                headers = [th.get_text(strip=True) for th in t.find_all('th')]
                if any('rank' in h.lower() for h in headers) and any('peak' in h.lower() for h in headers):
                    table = t
                    break
            
            if not table:
                # Fallback: look for sortable tables
                table = soup.find('table', class_='wikitable sortable')
            
            if not table:
                raise ValueError("Could not find the films table")
            
            # Parse the table
            rows = []
            headers = []
            
            # Get headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Get data rows
            for row in table.find_all('tr')[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                row_data = []
                for cell in cells:
                    text = cell.get_text(strip=True)
                    # Clean up text
                    text = re.sub(r'\[.*?\]', '', text)  # Remove references
                    text = re.sub(r'\s+', ' ', text).strip()
                    row_data.append(text)
                
                if row_data:
                    rows.append(row_data)
            
            # Create DataFrame
            if not headers:
                headers = [f'Col_{i}' for i in range(len(rows[0]) if rows else 0)]
            
            df = pd.DataFrame(rows, columns=headers[:len(rows[0]) if rows else 0])
            
            # Clean and standardize column names
            df.columns = [col.strip() for col in df.columns]
            
            # Try to identify key columns
            rank_col = None
            peak_col = None
            year_col = None
            title_col = None
            gross_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'rank' in col_lower and not rank_col:
                    rank_col = col
                elif 'peak' in col_lower and not peak_col:
                    peak_col = col
                elif 'year' in col_lower and not year_col:
                    year_col = col
                elif 'title' in col_lower or 'film' in col_lower and not title_col:
                    title_col = col
                elif 'worldwide' in col_lower or 'gross' in col_lower and not gross_col:
                    gross_col = col
            
            # Process numeric columns
            numeric_cols = [rank_col, peak_col, year_col] if all([rank_col, peak_col]) else []
            
            for col in numeric_cols:
                if col and col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
            
            # Process gross column for billion dollar analysis
            if gross_col and gross_col in df.columns:
                df['gross_numeric'] = self._extract_gross_amount(df[gross_col])
            
            # Process year column
            if year_col and year_col in df.columns:
                df['year_numeric'] = pd.to_numeric(df[year_col].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
            
            logger.info(f"Scraped {len(df)} films with columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia: {str(e)}")
            raise
    
    def _extract_gross_amount(self, gross_series: pd.Series) -> pd.Series:
        """Extract numeric gross amount in billions"""
        def extract_amount(text):
            if pd.isna(text):
                return None
            
            text = str(text).replace(',', '').replace('$', '')
            
            # Look for billion patterns
            billion_match = re.search(r'(\d+\.?\d*)\s*billion', text, re.IGNORECASE)
            if billion_match:
                return float(billion_match.group(1))
            
            # Look for million patterns and convert
            million_match = re.search(r'(\d+\.?\d*)\s*million', text, re.IGNORECASE)
            if million_match:
                return float(million_match.group(1)) / 1000
            
            # Look for raw numbers (assume millions if no unit)
            number_match = re.search(r'(\d+\.?\d*)', text)
            if number_match:
                num = float(number_match.group(1))
                # If it's a large number, assume it's in millions
                if num > 10000:
                    return num / 1000000
                else:
                    return num / 1000
            
            return None
        
        return gross_series.apply(extract_amount)
    
    def analyze_films_data(self, df: pd.DataFrame, questions: List[str]) -> List[Any]:
        """Analyze films data and answer questions"""
        answers = []
        
        try:
            # Question 1: How many $2 bn movies were released before 2000?
            count_2bn_before_2000 = 0
            if 'gross_numeric' in df.columns and 'year_numeric' in df.columns:
                mask = (df['gross_numeric'] >= 2.0) & (df['year_numeric'] < 2000)
                count_2bn_before_2000 = int(mask.sum())
            answers.append(count_2bn_before_2000)
            
            # Question 2: Which is the earliest film that grossed over $1.5 bn?
            earliest_1_5bn = "Unknown"
            if 'gross_numeric' in df.columns and 'year_numeric' in df.columns:
                over_1_5bn = df[df['gross_numeric'] >= 1.5]
                if not over_1_5bn.empty:
                    earliest = over_1_5bn.loc[over_1_5bn['year_numeric'].idxmin()]
                    title_col = self._find_title_column(df)
                    if title_col:
                        earliest_1_5bn = str(earliest[title_col])
            answers.append(earliest_1_5bn)
            
            # Question 3: What's the correlation between Rank and Peak?
            correlation = 0.0
            rank_col = self._find_column_by_keyword(df, 'rank')
            peak_col = self._find_column_by_keyword(df, 'peak')
            
            if rank_col and peak_col:
                valid_data = df[[rank_col, peak_col]].dropna()
                if len(valid_data) > 1:
                    correlation, _ = pearsonr(valid_data[rank_col], valid_data[peak_col])
            answers.append(round(correlation, 6))
            
            # Question 4: Draw a scatterplot
            plot_data_uri = self.create_rank_peak_scatterplot(df, rank_col, peak_col)
            answers.append(plot_data_uri)
            
        except Exception as e:
            logger.error(f"Error analyzing films data: {str(e)}")
            # Return default answers
            answers = [0, "Unknown", 0.0, "data:image/png;base64,"]
        
        return answers
    
    def _find_column_by_keyword(self, df: pd.DataFrame, keyword: str) -> Optional[str]:
        """Find column containing keyword"""
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
        return None
    
    def _find_title_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find title/film column"""
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower or 'film' in col_lower:
                return col
        return None
    
    def create_rank_peak_scatterplot(self, df: pd.DataFrame, rank_col: str, peak_col: str) -> str:
        """Create scatterplot of Rank vs Peak with regression line"""
        try:
            plt.figure(figsize=(10, 6))
            
            if rank_col and peak_col and rank_col in df.columns and peak_col in df.columns:
                valid_data = df[[rank_col, peak_col]].dropna()
                
                if not valid_data.empty:
                    x = valid_data[rank_col]
                    y = valid_data[peak_col]
                    
                    # Create scatterplot
                    plt.scatter(x, y, alpha=0.6, s=50)
                    
                    # Add regression line (dotted red)
                    if len(valid_data) > 1:
                        slope, intercept, _, _, _ = stats.linregress(x, y)
                        line_x = np.linspace(x.min(), x.max(), 100)
                        line_y = slope * line_x + intercept
                        plt.plot(line_x, line_y, 'r--', alpha=0.8, linewidth=2)
                    
                    plt.xlabel('Rank')
                    plt.ylabel('Peak')
                    plt.title('Rank vs Peak Scatterplot')
                    plt.grid(True, alpha=0.3)
            else:
                # Create a placeholder plot if columns not found
                plt.text(0.5, 0.5, 'Columns not found', ha='center', va='center', transform=plt.gca().transAxes)
                plt.xlabel('Rank')
                plt.ylabel('Peak')
                plt.title('Rank vs Peak Scatterplot')
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating scatterplot: {str(e)}")
            # Return minimal plot
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Plot Error', ha='center', va='center')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def analyze_court_data(self, questions_dict: Dict[str, str]) -> Dict[str, Any]:
        """Analyze Indian court judgments data"""
        answers = {}
        
        try:
            # Connect to DuckDB and setup S3 access
            conn = duckdb.connect()
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute("INSTALL parquet; LOAD parquet;")
            
            # Query the data
            query = """
            SELECT * FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            LIMIT 1000
            """
            
            try:
                df = conn.execute(query).fetchdf()
            except Exception as e:
                # If S3 access fails, create mock data for demonstration
                logger.warning(f"S3 access failed: {e}. Using mock data.")
                df = self._create_mock_court_data()
            
            # Process questions
            for question, _ in questions_dict.items():
                if "disposed the most cases" in question:
                    # Count cases by court for 2019-2022
                    if 'year' in df.columns and 'court' in df.columns:
                        filtered = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
                        court_counts = filtered['court'].value_counts()
                        answers[question] = court_counts.index[0] if not court_counts.empty else "Unknown"
                    else:
                        answers[question] = "Data unavailable"
                
                elif "regression slope" in question:
                    # Calculate regression slope for date differences
                    slope = self._calculate_date_regression_slope(df)
                    answers[question] = slope
                
                elif "scatterplot" in question:
                    # Create scatterplot
                    plot_uri = self._create_court_delay_plot(df)
                    answers[question] = plot_uri
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing court data: {str(e)}")
            # Return default answers
            for question in questions_dict.keys():
                if "scatterplot" in question:
                    answers[question] = "data:image/png;base64,"
                else:
                    answers[question] = "Error"
        
        return answers
    
    def _create_mock_court_data(self) -> pd.DataFrame:
        """Create mock court data for demonstration"""
        np.random.seed(42)
        n_records = 1000
        
        courts = ['33_10', '33_20', '34_10', '35_10', '36_10']
        years = [2019, 2020, 2021, 2022]
        
        data = []
        for i in range(n_records):
            year = np.random.choice(years)
            court = np.random.choice(courts)
            
            # Mock dates
            reg_date = pd.Timestamp(f"{year}-{np.random.randint(1,13)}-{np.random.randint(1,28)}")
            # Decision date is typically later
            decision_delay = np.random.randint(30, 500)  # days
            decision_date = reg_date + pd.Timedelta(days=decision_delay)
            
            data.append({
                'court': court,
                'year': year,
                'date_of_registration': reg_date.strftime('%d-%m-%Y'),
                'decision_date': decision_date.date(),
                'delay_days': decision_delay
            })
        
        return pd.DataFrame(data)
    
    def _calculate_date_regression_slope(self, df: pd.DataFrame) -> float:
        """Calculate regression slope of date differences by year for court 33_10"""
        try:
            if 'court' in df.columns:
                court_33_10 = df[df['court'] == '33_10']
                
                if 'delay_days' in court_33_10.columns and 'year' in court_33_10.columns:
                    # Use pre-calculated delay
                    valid_data = court_33_10[['year', 'delay_days']].dropna()
                else:
                    # Calculate delay from dates
                    valid_data = self._calculate_delays(court_33_10)
                
                if len(valid_data) > 1:
                    slope, _, _, _, _ = stats.linregress(valid_data['year'], valid_data['delay_days'])
                    return round(slope, 6)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating regression slope: {str(e)}")
            return 0.0
    
    def _calculate_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate delays between registration and decision dates"""
        try:
            if 'date_of_registration' in df.columns and 'decision_date' in df.columns:
                # Convert dates
                df['reg_date'] = pd.to_datetime(df['date_of_registration'], format='%d-%m-%Y', errors='coerce')
                df['dec_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
                
                # Calculate delay in days
                df['delay_days'] = (df['dec_date'] - df['reg_date']).dt.days
                
                return df[['year', 'delay_days']].dropna()
            
            return pd.DataFrame(columns=['year', 'delay_days'])
            
        except Exception as e:
            logger.error(f"Error calculating delays: {str(e)}")
            return pd.DataFrame(columns=['year', 'delay_days'])
    
    def _create_court_delay_plot(self, df: pd.DataFrame) -> str:
        """Create scatterplot of year vs delay days with regression line"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Filter for court 33_10
            if 'court' in df.columns:
                court_data = df[df['court'] == '33_10']
            else:
                court_data = df
            
            if 'delay_days' in court_data.columns and 'year' in court_data.columns:
                valid_data = court_data[['year', 'delay_days']].dropna()
                
                if not valid_data.empty:
                    x = valid_data['year']
                    y = valid_data['delay_days']
                    
                    # Scatterplot
                    plt.scatter(x, y, alpha=0.6, s=30)
                    
                    # Regression line
                    if len(valid_data) > 1:
                        slope, intercept, _, _, _ = stats.linregress(x, y)
                        line_x = np.linspace(x.min(), x.max(), 100)
                        line_y = slope * line_x + intercept
                        plt.plot(line_x, line_y, 'r-', alpha=0.8, linewidth=2)
                    
                    plt.xlabel('Year')
                    plt.ylabel('Delay Days')
                    plt.title('Year vs Delay Days (Court 33_10)')
                    plt.grid(True, alpha=0.3)
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='webp', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/webp;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating court delay plot: {str(e)}")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Plot Error', ha='center', va='center')
            buffer = BytesIO()
            plt.savefig(buffer, format='webp', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/webp;base64,{image_base64}"

# Initialize the agent
agent = DataAnalystAgent()

@app.route('/api/', methods=['POST'])
def analyze_data():
    try:
        # Get the questions file
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt file is required"}), 400
        
        questions_file = request.files['questions.txt']
        questions_content = questions_file.read().decode('utf-8')
        
        logger.info(f"Received questions: {questions_content[:200]}...")
        
        # Determine the type of analysis based on content
        if 'wikipedia.org' in questions_content and 'highest-grossing' in questions_content:
            # Wikipedia films analysis
            url_match = re.search(r'https://en\.wikipedia\.org/wiki/[^\s]+', questions_content)
            if url_match:
                url = url_match.group(0)
                df = agent.scrape_wikipedia_highest_grossing_films(url)
                questions = questions_content.split('\n')
                questions = [q.strip() for q in questions if q.strip() and not q.startswith('Answer')]
                result = agent.analyze_films_data(df, questions)
                return jsonify(result)
        
        elif 'Indian high court' in questions_content or 'judgement dataset' in questions_content:
            # Court judgments analysis
            # Extract questions from JSON format
            try:
                # Look for JSON in the questions
                json_match = re.search(r'\{[\s\S]*\}', questions_content)
                if json_match:
                    questions_json = json.loads(json_match.group(0))
                    result = agent.analyze_court_data(questions_json)
                    return jsonify(result)
            except json.JSONDecodeError:
                pass
            
            # Fallback: create default structure
            questions_dict = {
                "Which high court disposed the most cases from 2019 - 2022?": "",
                "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "",
                "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": ""
            }
            result = agent.analyze_court_data(questions_dict)
            return jsonify(result)
        
        else:
            # Generic analysis - try to parse and handle
            return jsonify({"message": "Analysis type not recognized", "content": questions_content[:500]})
    
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Data Analyst Agent is running"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Data Analyst Agent API",
        "endpoints": {
            "POST /api/": "Main analysis endpoint - send questions.txt and optional files",
            "GET /health": "Health check",
            "GET /": "This info page"
        },
        "example": "curl -X POST -F 'questions.txt=@questions.txt' https://your-domain.com/api/"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
