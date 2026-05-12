import os
from openai import AsyncOpenAI

MODEL = "gpt-5.4-nano-2026-03-17"
ENV_VAR = "OPENAI_API_KEY"
DISPLAY_NAME = "OpenAI (gpt-5.4-nano)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return AsyncOpenAI(api_key=api_key)


async def stream_response(client, context):
    stream = await client.responses.create(
        model=MODEL,
        input=context.get_messages(),
        stream=True,
    )
    full_response = ""
    async for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                full_response += delta
                print(delta, end="", flush=True)

    print()
    context.append("assistant", full_response)
