import json
import tiktoken


MODEL = "gpt-5.4-nano-2026-03-17"


class Context:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_messages(self):
        return self.messages

    def append(self, role: str, content: str):
        if role not in ("user", "assistant"):
            raise ValueError(f"Role {role} not recognized. Must be 'user' or 'assistant'.")
        self.messages.append({"role": role, "content": content})

    def prune(self, last_n: int):
        """Keep only the system prompt + last n messages."""
        if last_n <= 0:
            return
        self.messages = [self.messages[0]] + self.messages[-last_n:]

    def count_tokens(self) -> int:
        try:
            encoding = tiktoken.encoding_for_model(MODEL)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in self.messages:
            num_tokens += 4  # per-message overhead
            num_tokens += len(encoding.encode(message["content"]))
        num_tokens += 2  # assistant reply priming
        return num_tokens

    def serialize(self, filepath: str, provider_name: str = "", model_name: str = ""):
        """Write the conversation to a JSONL file.

        First line is metadata (provider/model), subsequent lines are messages.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            # Metadata line
            meta = {"meta": {"provider": provider_name, "model": model_name}}
            f.write(json.dumps(meta) + "\n")
            # Message lines
            for msg in self.messages:
                f.write(json.dumps(msg) + "\n")

    @classmethod
    def deserialize(cls, filepath: str):
        """Load a conversation from a JSONL file.

        Returns (context, metadata_dict) where metadata_dict contains
        provider and model info from the first line.
        """
        messages = []
        metadata = {"provider": "", "model": ""}
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if i == 0 and "meta" in obj:
                    metadata = obj["meta"]
                    continue
                messages.append(obj)

        if not messages or messages[0]["role"] != "system":
            raise ValueError("JSONL file must start with a system message.")

        ctx = cls(messages[0]["content"])
        for msg in messages[1:]:
            ctx.messages.append(msg)
        return ctx, metadata

    async def compact(self, client):
        """Summarize older messages into a single system message."""
        if len(self.messages) <= 3:
            return

        messages_to_summarize = self.messages[1:-2]
        if not messages_to_summarize:  # pragma: no cover
            return

        summary_request = [
            {"role": "system", "content": "Summarize the following conversation briefly, capturing the main points and context."}
        ]
        for m in messages_to_summarize:
            summary_request.append({"role": "user", "content": f"{m['role'].capitalize()}: {m['content']}"})

        response = await client.chat.completions.create(
            model=MODEL,
            messages=summary_request,
            temperature=0.5,
        )

        summary = response.choices[0].message.content
        self.messages = [
            self.messages[0],
            {"role": "system", "content": f"Summary of earlier conversation: {summary}"},
            *self.messages[-2:],
        ]
