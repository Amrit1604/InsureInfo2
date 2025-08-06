# 🆓 FREE DEPLOYMENT OPTIONS
## No Payment Required - 100% Free Platforms

## 🎯 **RECOMMENDED: Vercel (Easiest & Free)**

### Why Vercel?
- ✅ **100% Free** for projects like yours
- ✅ **Automatic HTTPS** 
- ✅ **GitHub integration**
- ✅ **Global CDN**
- ✅ **No credit card required**

### Quick Deploy:
1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   cd your-project-folder
   vercel --prod
   ```

3. **Add Environment Variables** in Vercel dashboard:
   ```
   GOOGLE_API_KEY=AIzaSyAbbapE0edChzDmtx6aNQ3AO3ZQk_N0iO8
   GOOGLE_API_KEY_2=AIzaSyD4M2LUb2E-8qonzZ-trEchkVVpMRDaWKY
   GOOGLE_API_KEY_3=AIzaSyA48msij4SOOFrx6bUNQCYI09yFwJuUENg
   GOOGLE_API_KEY_4=AIzaSyDyphTWm7VSOgFlfrSQPpbmE6HOg6yxJy0
   ```

**Your API URL:** `https://your-project.vercel.app/hackrx/run`

---

## 🌟 **Alternative: Replit (Browser-Based)**

### Why Replit?
- ✅ **Completely free hosting**
- ✅ **No CLI needed** - works in browser
- ✅ **Always-on deployments**

### Steps:
1. Go to [replit.com](https://replit.com)
2. Import from GitHub: `Amrit1604/InsureInfo2`
3. Add environment variables in Secrets tab
4. Click "Deploy" 
5. Get your URL: `https://your-app.your-username.repl.co/hackrx/run`

---

## 🔗 **Alternative: Netlify Functions**

### Why Netlify?
- ✅ **Free tier** with generous limits
- ✅ **Serverless functions**
- ✅ **Automatic deployments**

### Steps:
1. Connect GitHub to [netlify.com](https://netlify.com)
2. Deploy from repository
3. Add environment variables
4. Functions auto-deploy

---

## 🏆 **RECOMMENDATION: Use Vercel!**

**Vercel is the fastest and most reliable free option for your hackathon API.**

### ⚡ **Quick Test After Deploy:**
```bash
curl -X POST "https://your-project.vercel.app/hackrx/run" \
  -H "Authorization: Bearer test-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": ["What is the grace period?"]
  }'
```

**All these platforms are 100% free - no payment required!** 🎉
