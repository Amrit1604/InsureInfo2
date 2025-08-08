import requests

keys = [
    # add your API keys here
    "AIzaSyA-kRkmjbNwe6fTCSR7x272jxxkQyVI6vE",
    "AIzaSyAbbapE0edChzDmtx6aNQ3AO3ZQk_N0iO8",
    "AIzaSyD4M2LUb2E-8qonzZ-trEchkVVpMRDaWKY",
    "AIzaSyA48msij4SOOFrx6bUNQCYI09yFwJuUENg",
    "AIzaSyDyphTWm7VSOgFlfrSQPpbmE6HOg6yxJy0",
    "AIzaSyAn3rz9yqyyckoNDfUSLPkyU8zl6ry14y8",
    "AIzaSyA4KaYpw9rCCr9tiNoXGwHEu8jAnf0HqDw",
    "AIzaSyD255YaJvsPJ_K3jKINJn3eAy_PI59X0Jo",
    "AIzaSyDccCSqK7XojK9dMezABR_credy3gTCKO0"
]

for key in keys:
    print(f"Testing key: {key}")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
        f"?key={key}"
    )
    payload = {
        "contents": [{"parts": [{"text": "Hello"}]}]
    }
    try:
        resp = requests.post(url, json=payload)
        status = resp.status_code
        data = resp.json()
        if status == 200 and "error" not in data:
            print("✅ VALID key\n")
        else:
            err = data.get("error", {}).get("message", "Unknown error")
            print(f"❌ Invalid/Denied (HTTP {status}, {err})\n")
    except Exception as e:
        print(f"⚠️ Request failed: {e}\n")
