# Deployment Guide - TDS Data Analyst Agent

This guide covers multiple deployment options for your Data Analyst Agent API.

## 🚀 Quick Deployment Options

### 1. Railway (Recommended) ⭐

**Why Railway?** 
- Automatic Python/Docker detection
- Zero configuration needed
- Built-in SSL/TLS
- Easy GitHub integration

**Steps:**
1. Push your code to GitHub
2. Go to [Railway.app](https://railway.app)
3. Sign up with GitHub
4. Click "Deploy from GitHub repo"
5. Select your `tds-data-analyst-agent` repository
6. Railway automatically detects the Dockerfile and deploys
7. Get your public URL (e.g., `https://your-app.railway.app`)

**Environment Variables (if needed):**
```
PORT=8000
FLASK_ENV=production
```

### 2. Render

**Steps:**
1. Go to [Render.com](https://render.com)
2. Connect your GitHub account
3. Create new "Web Service"
4. Select your repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Environment**: Python 3

### 3. Google Cloud Run

**Prerequisites**: Google Cloud account with billing enabled

```bash
# Install Google Cloud SDK
# Set up authentication
gcloud auth login
gcloud config set project YOUR-PROJECT-ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/data-analyst-agent

# Deploy to Cloud Run
gcloud run deploy data-analyst-agent \
  --image gcr.io/YOUR-PROJECT-ID/data-analyst-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3m
```

### 4. Heroku

```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create new app
heroku create your-data-analyst-agent

# Deploy
git push heroku main

# Scale up if needed
heroku ps:scale web=1
```

### 5. AWS ECS/Fargate

1. **Push to ECR:**
```bash
aws ecr create-repository --repository-name data-analyst-agent
docker tag data-analyst-agent:latest YOUR-ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/data-analyst-agent:latest
docker push YOUR-ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/data-analyst-agent:latest
```

2. **Create ECS Task Definition** (use AWS Console or CLI)
3. **Deploy as Fargate Service**

## 🧪 Local Testing

### Method 1: Python directly
```bash
python app.py
```

### Method 2: Docker
```bash
docker build -t data-analyst-agent .
docker run -p 8000:8000 data-analyst-agent
```

### Method 3: Using ngrok (for temporary public URL)
```bash
# Start the app locally
python app.py

# In another terminal
ngrok http 8000
# Use the ngrok URL for testing
```

## ✅ Testing Your Deployment

### 1. Health Check
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Data Analyst Agent is running"
}
```

### 2. Test Wikipedia Analysis
Create `test_questions.txt`:
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI under 100,000 bytes.
```

Test it:
```bash
curl -X POST -F "questions.txt=@test_questions.txt" https://your-domain.com/api/
```

### 3. Test Court Data Analysis
Create `court_questions.txt`:
```
The Indian high court judgement dataset contains judgements from the Indian High Courts.

Answer the following questions and respond with a JSON object containing the answer.

{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```

Test it:
```bash
curl -X POST -F "questions.txt=@court_questions.txt" https://your-domain.com/api/
```

## 🔧 Configuration Options

### Environment Variables
```bash
PORT=8000                    # Server port
FLASK_ENV=production         # Flask environment
LOG_LEVEL=INFO              # Logging level
MAX_CONTENT_LENGTH=100MB    # Max upload size
WORKERS=2                   # Gunicorn workers
TIMEOUT=180                 # Request timeout (seconds)
```

### Docker Build Args
```dockerfile
# For custom configurations
docker build --build-arg PYTHON_VERSION=3.11 -t data-analyst-agent .
```

## 📊 Performance Tuning

### For High Traffic
```bash
# Increase workers and memory
gunicorn --bind 0.0.0.0:8000 --workers 4 --worker-class gevent --worker-connections 1000 --timeout 180 app:app
```

### Resource Requirements
- **Memory**: Minimum 1GB, Recommended 2GB
- **CPU**: 1 vCPU minimum
- **Disk**: 1GB for dependencies
- **Network**: Outbound access for scraping

## 🛡️ Production Considerations

### Security
- Enable HTTPS (most platforms provide this automatically)
- Set up rate limiting if needed
- Monitor for suspicious activity
- Keep dependencies updated

### Monitoring
- Set up health check monitoring
- Log analysis and error tracking
- Performance metrics collection
- Uptime monitoring

### Scaling
- Use multiple worker processes
- Consider load balancing for high traffic
- Database connection pooling if needed
- Caching layer for repeated queries

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Ensure Dockerfile syntax is correct

2. **Memory Issues**
   - Increase memory allocation on platform
   - Optimize data processing in chunks
   - Use efficient DataFrame operations

3. **Timeout Errors**
   - Increase timeout settings
   - Optimize scraping and analysis code
   - Use async processing where possible

4. **Network Issues**
   - Verify outbound network access
   - Check firewall settings
   - Test external URL accessibility

### Debug Commands
```bash
# Check logs (Railway)
railway logs

# Check logs (Heroku)
heroku logs --tail

# Check logs (Docker)
docker logs CONTAINER_ID

# Local debug mode
FLASK_ENV=development python app.py
```

## 📋 Deployment Checklist

Before going live:

- [ ] Code pushed to GitHub repository
- [ ] All dependencies listed in requirements.txt
- [ ] Dockerfile builds successfully
- [ ] Local testing completed
- [ ] Health check endpoint responds correctly
- [ ] Sample analysis requests work
- [ ] Response format matches requirements
- [ ] Images generate correctly and stay under 100KB
- [ ] Error handling works gracefully
- [ ] Logging configured properly

## 📤 Submission

Once deployed:

1. **Test your API thoroughly** with the provided sample questions
2. **Note your URLs**:
   - GitHub Repository: `https://github.com/YOUR_USERNAME/tds-data-analyst-agent`
   - API Endpoint: `https://your-app.railway.app/api/` (or your platform's URL)
3. **Submit to**: `https://exam.sanand.workers.dev/tds-data-analyst-agent`

---

**🎯 You're ready to submit to the TDS challenge!**
