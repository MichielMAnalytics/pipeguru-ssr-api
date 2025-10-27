"""Simple test for LLM connection."""

import asyncio
from dotenv import load_dotenv
from src.core.llm_client import LLMClient

# Load environment variables from .env file
load_dotenv()

async def test_connection():
    print("=" * 60)
    print("TEST: OpenAI API Connection")
    print("=" * 60)

    try:
        client = LLMClient()
        print(f"✓ Client initialized with model: {client.model}")

        result = await client.test_connection()

        if result:
            print("✓ Connection successful!")
            print("✓ API key is valid")
        else:
            print("✗ Connection failed")

    except ValueError as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
