# 🚀 RENDER.COM DEPLOYMENT GUIDE
## Deploy Your Hackathon API to Production

### ✅ Pre-Deployment Checklist
- [x] `requirements.txt` ready
- [x] `runtime.txt` configured (Python 3.11)
- [x] `render.yaml` deployment config
- [x] `api_server.py` production-ready
- [x] Environment variables identified

### 🌐 **Step 1: Create Render Account**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Connect your GitHub repository

### 🔗 **Step 2: Connect Repository**
1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `Amrit1604/InsureInfo2`
3. Branch: `main`

### ⚙️ **Step 3: Configure Service**
```
Name: insure-info-hackathon-api
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: python api_server.py
Instance Type: Free
```

### 🔑 **Step 4: Set Environment Variables**
In Render Dashboard → Environment, add:
```
GOOGLE_API_KEY=your_first_google_api_key
GOOGLE_API_KEY_2=your_second_google_api_key
GOOGLE_API_KEY_3=your_third_google_api_key
GOOGLE_API_KEY_4=your_fourth_google_api_key
PORT=8000
HOST=0.0.0.0
```

### 🚀 **Step 5: Deploy**
1. Click "Create Web Service"
2. Wait 3-5 minutes for build
3. Your API will be live at: `https://your-service-name.onrender.com`

### 🏆 **Step 6: Submit to Hackathon**
**Webhook URL:** `https://your-service-name.onrender.com/hackrx/run`
**Description:** `FastAPI + Google Gemini 1.5 Flash + FAISS Vector Search + Ultra-Fast Caching`

### 🧪 **Step 7: Test Your Deployed API**
```bash
curl -X POST "https://your-service-name.onrender.com/hackrx/run" \
  -H "Authorization: Bearer hackrx_2025_test_key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

### 🔒 **Automatic HTTPS**
✅ Render automatically provides HTTPS certificates
✅ All requests are secure by default
✅ No additional configuration needed

### 📊 **Monitoring**
- Health check: `https://your-service-name.onrender.com/health`
- Logs: Available in Render dashboard
- Auto-deploy: Enabled for main branch

### 🏆 **Hackathon Advantages**
- ⚡ Free hosting with HTTPS
- 🔄 Auto-deploy from GitHub
- 📈 Built-in monitoring
- 🌍 Global CDN
- ⚡ Lightning fast deployment
