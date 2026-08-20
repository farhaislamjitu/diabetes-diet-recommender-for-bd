import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file!")
else:
    print("API key loaded successfully.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Say hello in one short sentence and confirm you are working."
    )

    print("\n=== Gemini Response ===")
    print(response.text)