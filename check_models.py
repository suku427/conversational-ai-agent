import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

print("🔍 Checking available Gemini models for your API key...")
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    for model in client.models.list():
        if hasattr(model, "name"):
            print(f"- {model.name}")
except Exception as e:
    print(f"❌ Error: {e}")