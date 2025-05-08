from openai import OpenAI
from loguru import logger
import requests
import json

def test_api():
    print("Testing API connection...")

    # First, let's check if the API server is running
    try:
        response = requests.get("http://localhost:7912/health")
        print(f"API health check: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"API health check failed: {e}")

    # Now try with the OpenAI client
    client = OpenAI(
        api_key='sk-local',
        base_url='http://localhost:7912'
    )

    try:
        print("Sending request to API...")
        response = client.chat.completions.create(
            model="claude-3-7-sonnet-20250219",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
            ],
            temperature=0
        )

        print(f"Response type: {type(response)}")
        print(f"Response: {response}")

        if response and hasattr(response, 'choices') and response.choices:
            print("API response successful!")
            print(f"Content: {response.choices[0].message.content}")
            return True
        else:
            print(f"API returned an invalid response: {response}")
            return False
    except Exception as e:
        print(f"Error calling API: {e}")
        return False

if __name__ == "__main__":
    test_api()
