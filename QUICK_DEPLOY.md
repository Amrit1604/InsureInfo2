# 🚀 Quick Deployment Guide

## ⚡ **Fastest Option: Replit (RECOMMENDED)**

### **Step 1: Import to Replit**
1. Go to [replit.com](https://replit.com)
2. Click **"Create Repl"**
3. Select **"Import from GitHub"**
4. URL: `https://github.com/Amrit1604/InsureInfo2`
5. Click **"Import"**

### **Step 2: Add Environment Variables**
Click the **🔒 Secrets** tab and add:
```
GOOGLE_API_KEY=your-primary-gemini-api-key
GOOGLE_API_KEY_PRO=your-secondary-gemini-api-key
GOOGLE_API_KEY_3=your-tertiary-gemini-api-key
GOOGLE_API_KEY_4=your-quaternary-gemini-api-key
```

### **Step 3: Run**
1. Click **"Run"** button
2. Wait for dependencies to install (2-3 minutes)
3. Your API will be live at: `https://your-repl-name.your-username.repl.co`

### **Step 4: Test Hackathon Endpoint**
Your hackathon endpoint will be:
```
https://your-repl-name.your-username.repl.co/hackrx/run
```

---

## 🔄 **Alternative: Netlify**

### **Step 1: Connect Repository**
1. Go to [netlify.com](https://netlify.com)
2. Click **"New site from Git"**
3. Connect your GitHub: `https://github.com/Amrit1604/InsureInfo2`

### **Step 2: Configure Build**
- **Build command**: `pip install -r requirements.txt`
- **Publish directory**: `.`

### **Step 3: Add Environment Variables**
In Site Settings → Environment Variables, add your Google API keys.

---

## 🎯 **Why Replit is Best for This Project**

✅ **Zero configuration** - Works out of the box
✅ **ML packages supported** - Handles torch, transformers, faiss
✅ **Automatic HTTPS** - No SSL setup needed
✅ **Always-on hosting** - Keeps your API running
✅ **Fast deployment** - 5 minutes from start to finish
✅ **Free tier sufficient** - Perfect for hackathons

---

## 🏆 **After Deployment**

### **Test Your API**
```bash
curl -X POST "https://your-deployment-url/hackrx/run" \
  -H "Authorization: Bearer hackrx_2025_insure_key_001" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

### **Expected Response**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days from the due date..."
  ]
}
```

---

## 🚨 **Troubleshooting**

### **If Replit fails to start:**
1. Open the **Shell** tab
2. Run: `pip install -r requirements.txt`
3. Run: `python api_server.py`

### **If you need more API keys:**
Add them as environment variables in Replit Secrets.

---

**🎯 Your hackathon-ready API with HTTPS is just 5 minutes away!**
