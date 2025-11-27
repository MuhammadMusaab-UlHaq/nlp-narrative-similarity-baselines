import os
import openai
import sys

# Setup OpenAI Client
API_KEY = os.getenv("POE_API_KEY")
if not API_KEY:
    print("POE_API_KEY not set")
    sys.exit(1)

print(f"Testing Poe API with Key: {API_KEY[:5]}...")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.poe.com/v1",
)

print("Attempting simple chat completion...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, are you working?"}],
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error without json_object: {e}")

print("\nAttempting chat completion with response_format={'type': 'json_object'}...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Output JSON."},
            {"role": "user", "content": "Output a JSON object with key 'status' and value 'ok'."}
        ],
        response_format={"type": "json_object"}
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error with json_object: {e}")
