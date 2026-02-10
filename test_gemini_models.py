"""Test script to check available Gemini models."""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in environment")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:")
print("-" * 60)

try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Supported Methods: {model.supported_generation_methods}")
            print()
except Exception as e:
    print(f"Error listing models: {e}")

# Try to use a model
print("\nTesting model initialization:")
print("-" * 60)

test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"✓ {model_name} - Successfully initialized")
        
        # Try a simple generation
        response = model.generate_content("Say 'test'")
        print(f"  Response: {response.text[:50]}")
    except Exception as e:
        print(f"✗ {model_name} - Failed: {e}")
