"""
Setup instructions for Sathik AI Direction Mode
Complete installation and configuration guide
"""

# 1. Install Dependencies

# For Python backend
pip install -r requirements_direction_mode.txt

# For React frontend
cd web_ui
npm install

# 2. Configure API Keys (Optional)

# Edit config_direction_mode.py to add API keys:
DIRECTION_MODE_CONFIG = {
    'google_api_key': 'your-google-api-key',
    'google_cse_id': 'your-google-cse-id', 
    'news_api_key': 'your-news-api-key',
    # ... other config
}

# 3. Environment Variables (Optional)

# Create .env file for API keys:
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-google-cse-id
NEWS_API_KEY=your-news-api-key
API_KEY=your-optional-api-key
ALLOWED_ORIGINS=*

# 4. Run the System

# Option A: Terminal Interface
python main.py --mode terminal

# Option B: FastAPI Server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Option C: Development (both backend and frontend)
# Terminal 1: Backend
python main.py --mode terminal

# Terminal 2: Frontend  
cd web_ui
npm run dev

# 5. Access the Web Interface

# Open browser to: http://localhost:3000
# FastAPI docs: http://localhost:8000/docs

# 6. Docker Deployment (Optional)

# Build and run with Docker
docker-compose up --build

# 7. Usage Examples

# In terminal interface:
mode direction          # Switch to Direction Mode
submode sugarcotted    # Change to sweet style
What is artificial intelligence?  # Ask a question

# In web interface:
# - Select Direction Mode
# - Choose sub-mode (Sugarcotted, Unhinged, Reaper, 666)
# - Enter query and submit

# 8. API Usage

# REST API endpoints:
POST /query - Submit query
GET /modes - Get available modes
GET /stats - Get system statistics
GET /status - Get system status
POST /clear-cache - Clear cache
POST /search-knowledge - Search knowledge base

# 9. Troubleshooting

# Common issues:
# 1. Import errors: Ensure all dependencies installed
# 2. API key errors: Check configuration files
# 3. Port conflicts: Change ports in config files
# 4. CORS errors: Check allowed origins in security config

# Debug commands:
python -c "from sathik_ai.direction_mode import DirectionModeController; print('Import successful')"
python main.py --mode terminal --debug