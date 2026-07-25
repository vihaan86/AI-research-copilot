import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

def load_client() -> genai.Client:
    """Create the Gemini client."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    return genai.Client(
        api_key=api_key
    )

def generate_response(
    prompt: str,
    client: genai.Client,
) -> str:
    """Generate a response from Gemini."""

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":
    client = load_client()

    prompt = input("Enter your prompt: ")

    answer = generate_response(
        prompt,
        client,
    )

    print("\nResponse:\n")
    print(answer)
