import os

from google import genai


def generate_ml_sentence(prompt: str = "Napisz jedno zdanie o uczeniu maszynowym.") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    return response.text


if __name__ == "__main__":
    print(generate_ml_sentence())
