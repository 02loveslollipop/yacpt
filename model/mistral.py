import os
from mistralai.client import Mistral

MODEL = "mistral-small-latest"
ENV_VAR = "MISTRAL_API_KEY"
DISPLAY_NAME = "Mistral (mistral-small)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return Mistral(api_key=api_key)


async def stream_response(client, context):
    messages = context.get_messages()

    full_response = ""
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=messages,
    )
    async for chunk in stream:
        delta = chunk.data.choices[0].delta.content
        if delta:
            full_response += delta
            print(delta, end="", flush=True)

    print()
    context.append("assistant", full_response)
