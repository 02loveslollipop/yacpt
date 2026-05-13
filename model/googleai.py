import os
from google import genai
from google.genai import types

MODEL = "gemma-4-31b-it"
ENV_VAR = "GOOGLE_API_KEY"
DISPLAY_NAME = "Google (Gemma 4 31B)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return genai.Client(api_key=api_key)


async def stream_response(client, context):
    messages = context.get_messages()

    # Extract system instruction and convert message format
    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            if system_instruction is None:
                system_instruction = msg["content"]
            else:
                system_instruction += "\n" + msg["content"]
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    full_response = ""
    stream = await client.aio.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    async for chunk in stream:
        if chunk.text:
            full_response += chunk.text
            print(chunk.text, end="", flush=True)

    print()
    context.append("assistant", full_response)
