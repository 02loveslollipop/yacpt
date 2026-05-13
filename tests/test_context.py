"""Unit tests for classes/context.py — targeting 100% coverage."""
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from classes.context import Context


# ── __init__ ──────────────────────────────────────────────────────────────

class TestContextInit:
    def test_creates_system_message(self):
        ctx = Context("Hello system")
        msgs = ctx.get_messages()
        assert len(msgs) == 1
        assert msgs[0] == {"role": "system", "content": "Hello system"}


# ── get_messages ──────────────────────────────────────────────────────────

class TestGetMessages:
    def test_returns_messages_list(self, mock_context):
        msgs = mock_context.get_messages()
        assert isinstance(msgs, list)
        assert len(msgs) == 3  # system + user + assistant


# ── append ────────────────────────────────────────────────────────────────

class TestAppend:
    def test_append_user(self):
        ctx = Context("sys")
        ctx.append("user", "hi")
        assert ctx.get_messages()[-1] == {"role": "user", "content": "hi"}

    def test_append_assistant(self):
        ctx = Context("sys")
        ctx.append("assistant", "hello")
        assert ctx.get_messages()[-1] == {"role": "assistant", "content": "hello"}

    def test_append_invalid_role_raises(self):
        ctx = Context("sys")
        with pytest.raises(ValueError, match="Role invalid not recognized"):
            ctx.append("invalid", "nope")


# ── prune ─────────────────────────────────────────────────────────────────

class TestPrune:
    def test_prune_keeps_system_plus_last_n(self, mock_context):
        mock_context.append("user", "msg2")
        mock_context.append("assistant", "reply2")
        mock_context.prune(2)
        msgs = mock_context.get_messages()
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 3  # system + last 2

    def test_prune_zero_is_noop(self, mock_context):
        original_len = len(mock_context.get_messages())
        mock_context.prune(0)
        assert len(mock_context.get_messages()) == original_len

    def test_prune_negative_is_noop(self, mock_context):
        original_len = len(mock_context.get_messages())
        mock_context.prune(-1)
        assert len(mock_context.get_messages()) == original_len


# ── count_tokens ──────────────────────────────────────────────────────────

class TestCountTokens:
    def test_returns_positive_int(self, mock_context):
        tokens = mock_context.count_tokens()
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_fallback_encoding_on_key_error(self):
        """Exercise the KeyError branch by using an unknown model name."""
        ctx = Context("test")
        with patch("classes.context.MODEL", "nonexistent-model-xyz"):
            tokens = ctx.count_tokens()
            assert tokens > 0


# ── serialize / deserialize ───────────────────────────────────────────────

class TestSerialize:
    def test_writes_metadata_and_messages(self, mock_context, tmp_path):
        filepath = str(tmp_path / "conv.jsonl")
        mock_context.serialize(filepath, provider_name="openai", model_name="gpt-5")
        with open(filepath) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert lines[0] == {"meta": {"provider": "openai", "model": "gpt-5"}}
        assert lines[1]["role"] == "system"
        assert lines[2]["role"] == "user"
        assert lines[3]["role"] == "assistant"
        assert len(lines) == 4

    def test_default_empty_metadata(self, tmp_path):
        ctx = Context("sys")
        filepath = str(tmp_path / "conv.jsonl")
        ctx.serialize(filepath)
        with open(filepath) as f:
            meta = json.loads(f.readline())
        assert meta == {"meta": {"provider": "", "model": ""}}


class TestDeserialize:
    def test_round_trip(self, mock_context, tmp_path):
        filepath = str(tmp_path / "conv.jsonl")
        mock_context.serialize(filepath, provider_name="mistral", model_name="small")
        ctx2, meta = Context.deserialize(filepath)
        assert meta == {"provider": "mistral", "model": "small"}
        assert len(ctx2.get_messages()) == len(mock_context.get_messages())
        assert ctx2.get_messages()[0]["content"] == mock_context.get_messages()[0]["content"]

    def test_deserialize_from_sample(self, sample_jsonl):
        ctx, meta = Context.deserialize(str(sample_jsonl))
        assert meta["provider"] == "googleai"
        assert meta["model"] == "gemma-4-31b-it"
        assert len(ctx.get_messages()) == 3

    def test_missing_system_raises(self, tmp_path):
        filepath = tmp_path / "bad.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"meta": {"provider": "", "model": ""}}) + "\n")
            f.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
        with pytest.raises(ValueError, match="must start with a system message"):
            Context.deserialize(str(filepath))

    def test_empty_messages_raises(self, tmp_path):
        filepath = tmp_path / "empty.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"meta": {"provider": "", "model": ""}}) + "\n")
        with pytest.raises(ValueError, match="must start with a system message"):
            Context.deserialize(str(filepath))

    def test_skips_empty_lines(self, tmp_path):
        filepath = tmp_path / "gaps.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"meta": {"provider": "", "model": ""}}) + "\n")
            f.write("\n")  # empty line
            f.write(json.dumps({"role": "system", "content": "sys"}) + "\n")
            f.write("\n")
            f.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
        ctx, _ = Context.deserialize(str(filepath))
        assert len(ctx.get_messages()) == 2

    def test_no_meta_line(self, tmp_path):
        """File without a meta line — first line is the system message."""
        filepath = tmp_path / "nometa.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"role": "system", "content": "sys"}) + "\n")
            f.write(json.dumps({"role": "user", "content": "hi"}) + "\n")
        ctx, meta = Context.deserialize(str(filepath))
        assert meta == {"provider": "", "model": ""}
        assert len(ctx.get_messages()) == 2


# ── compact ───────────────────────────────────────────────────────────────

class TestCompact:
    @pytest.mark.asyncio
    async def test_compact_few_messages_noop(self):
        """≤ 3 messages should return without doing anything."""
        ctx = Context("sys")
        ctx.append("user", "hi")
        mock_client = AsyncMock()
        await ctx.compact(mock_client)
        assert len(ctx.get_messages()) == 2
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_four_messages_calls_api(self):
        """4 messages: len > 3 and messages[1:-2] has 1 item, so API is called."""
        ctx = Context("sys")
        ctx.append("user", "q1")
        ctx.append("assistant", "a1")
        ctx.append("user", "q2")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        await ctx.compact(mock_client)
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_summarizes(self):
        """Normal compaction with >3 messages."""
        ctx = Context("sys")
        ctx.append("user", "q1")
        ctx.append("assistant", "a1")
        ctx.append("user", "q2")
        ctx.append("assistant", "a2")
        # 5 messages. messages_to_summarize = messages[1:-2] = [user:q1, assistant:a1]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of conversation"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await ctx.compact(mock_client)

        msgs = ctx.get_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "system"
        assert "Summary" in msgs[1]["content"]
        assert len(msgs) == 4  # system + summary + last 2
