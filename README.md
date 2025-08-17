# TDS Data Analyst Agent

An intelligent API service that automatically sources, prepares, analyzes, and visualizes data using LLMs.

## 🌟 Features

- 🌐 **Web Scraping**: Wikipedia tables and other data sources
- 📊 **Data Processing**: CSV, Excel, database queries via DuckDB
- 📈 **Statistical Analysis**: Correlation, regression, aggregations
- 🎨 **Visualizations**: Charts as base64-encoded data URIs
- ⚡ **Fast Response**: Results within 3 minutes
- 🔧 **Production Ready**: Docker, health checks, comprehensive logging

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The server will start on `http://localhost:8000`

### Docker

```bash
# Build the image
docker build -t data-analyst-agent .

# Run the container
docker run -p 8000:8000 data-analyst-agent
```

## 📡 API Usage

### Main Endpoint: `POST /api/`

Send analysis requests with a `questions.txt` file and optional data files:

```bash
curl -X POST -F "questions.txt=@questions.txt" https://your-domain.com/api/
```

### Example Questions

#### Wikipedia Films Analysis
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
```

**Response**: `[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]`

#### Court Data Analysis
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```

## 🏗️ Architecture

- **Flask API** with RESTful endpoints
- **Beautiful Soup** for web scraping
- **Pandas/NumPy** for data processing
- **Matplotlib/Seaborn** for visualizations
- **DuckDB** for efficient database queries
- **Base64 encoding** for image responses

## 📊 Supported Analysis Types

1. **Wikipedia Data Scraping**
   - Automatic table extraction
   - Data cleaning and processing
   - Statistical analysis

2. **Database Analysis**
   - DuckDB with S3 integration
   - Large dataset processing
   - Complex aggregations

3. **File Processing**
   - CSV/Excel file analysis
   - Custom data transformations

## 🚀 Deployment

### Railway (Recommended)
1. Connect your GitHub repository
2. Railway auto-detects Python/Docker
3. Deploys automatically

### Other Platforms
- **Render**: Web service deployment
- **Google Cloud Run**: Containerized deployment
- **Heroku**: Traditional PaaS deployment

See `DEPLOYMENT.md` for detailed instructions.

## 🔍 Health Check

```bash
curl http://your-domain.com/health
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🐛 Troubleshooting

- **S3 Access Issues**: Falls back to mock data automatically
- **Scraping Failures**: Graceful error handling with default responses
- **Memory Issues**: Optimized for large dataset processing
- **Plot Generation**: Returns placeholder on visualization errors

Check logs for detailed error information.

## 📈 Performance

- Response timeout: 3 minutes
- Max file upload: 100MB
- Image size limit: 100KB (base64)
- Optimized for concurrent requests

---

Ready for the TDS Data Analyst Agent challenge! 🎯
