# EMERGENCY DEPLOYMENT - HEROKU ONE-CLICK
## Railway Failed Due to Image Size (7.7GB > 4GB limit)

### 🚨 **IMMEDIATE SOLUTION: Use Heroku**

**Click here for one-click deploy:**
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/Amrit1604/InsureInfo2)

### **Why Heroku Will Work:**
- ✅ **No image size limits** like Railway
- ✅ **ML-friendly** - handles large dependencies
- ✅ **Reliable** - used by millions of apps
- ✅ **Free tier** available

### **Steps:**
1. Click the Heroku deploy button above
2. Add your 4 Google API keys
3. Deploy in ~5-10 minutes
4. Get your URL: `https://your-app-name.herokuapp.com/hackrx/run`

### **Alternative Quick Fix for Railway:**
If you want to try Railway again, I can create a minimal version without heavy ML dependencies, but Heroku is more reliable for your use case.

### **Your API Endpoint Will Be:**
```
POST https://your-app-name.herokuapp.com/hackrx/run
Authorization: Bearer your_token
{
  "documents": "https://policy-url.pdf",
  "questions": ["Your questions"]
}
```

**Heroku is the safe bet for hackathon submission!** 🚀
