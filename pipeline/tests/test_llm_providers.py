"""Pluggable LLM backends.

The scoring stage asks a client three things — can you see images, how much
should I trust you, and give me JSON in this shape. These lock in that a
provider added later cannot quietly get those wrong.
"""

import json

import pytest

from publikclip_pipeline.scoring import llm


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Never read the developer's real secrets.json, and never write cache
    into their home."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    for provider in llm.PROVIDERS.values():
        monkeypatch.delenv(provider.key_env, raising=False)
    for var in (
        "PUBLIKCLIP_LLM_BASE_URL",
        "PUBLIKCLIP_LLM_MODEL",
        "PUBLIKCLIP_LLM_VISION_MODEL",
        "PUBLIKCLIP_LLM_API_KEY",
        "PUBLIKCLIP_LLM_STRUCTURED",
        "PUBLIKCLIP_LLM_LABEL",
    ):
        monkeypatch.delenv(var, raising=False)


SCHEMA = {"type": "object", "properties": {"hook": {"type": "integer"}}, "required": ["hook"]}


# --- dispatch -------------------------------------------------------------


def test_available_modes_lists_every_preset():
    modes = llm.available_modes()
    assert modes[:2] == ["gemini", "ollama"]
    for name in llm.PROVIDERS:
        assert name in modes


def test_custom_is_hidden_until_it_is_configured(monkeypatch):
    assert "custom" not in llm.available_modes()
    monkeypatch.setenv("PUBLIKCLIP_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("PUBLIKCLIP_LLM_MODEL", "my-model")
    assert "custom" in llm.available_modes()


def test_unknown_mode_raises_instead_of_falling_back_to_gemini():
    """A typo in --llm used to spend the Gemini quota silently."""
    with pytest.raises(llm.LlmError) as err:
        llm.make_client("gemni")
    assert "Unknown LLM mode" in str(err.value)


def test_missing_key_names_the_env_var_and_where_to_get_one():
    with pytest.raises(llm.LlmError) as err:
        llm.make_client("nvidia")
    message = str(err.value)
    assert "PUBLIKCLIP_NVIDIA_API_KEY" in message
    assert "build.nvidia.com" in message


def test_custom_mode_without_config_explains_what_is_missing():
    with pytest.raises(llm.LlmError) as err:
        llm.make_client("custom")
    assert "PUBLIKCLIP_LLM_BASE_URL" in str(err.value)


def test_env_key_beats_secrets_file(tmp_path, monkeypatch):
    (tmp_path / "secrets.json").write_text(json.dumps({"nvidia_api_key": "from-file"}))
    assert llm.make_client("nvidia")._key == "from-file"
    monkeypatch.setenv("PUBLIKCLIP_NVIDIA_API_KEY", "from-env")
    assert llm.make_client("nvidia")._key == "from-env"


def test_key_is_read_from_secrets_file(tmp_path):
    (tmp_path / "secrets.json").write_text(json.dumps({"nvidia_api_key": "  padded  "}))
    assert llm.make_client("nvidia")._key == "padded"


# --- capabilities the scoring stage reads ---------------------------------


def test_every_client_exposes_the_capabilities_the_stage_uses(monkeypatch):
    """scoring.stage asks for these by attribute; a provider missing one
    would crash a job halfway through rather than at startup."""
    monkeypatch.setenv("PUBLIKCLIP_NVIDIA_API_KEY", "x")
    for client in (llm.GeminiClient, llm.OllamaClient, llm.make_client("nvidia")):
        for attr in ("backend", "supports_vision", "confidence"):
            assert hasattr(client, attr), f"{client} lacks {attr}"


def test_confidence_labels_are_honest(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_NVIDIA_API_KEY", "x")
    assert llm.GeminiClient.confidence == "standard"
    assert llm.OllamaClient.confidence == "local-estimate"
    assert llm.make_client("nvidia").confidence == "third-party"


def test_provider_without_a_vision_model_reports_no_vision(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_GROQ_API_KEY", "x")
    client = llm.make_client("groq")
    assert client.supports_vision is False


# --- structured output flavours -------------------------------------------


def _client(monkeypatch, mode):
    monkeypatch.setenv(llm.PROVIDERS[mode].key_env, "x")
    return llm.make_client(mode)


def test_nvidia_uses_the_standard_response_format(monkeypatch):
    """Verified against the live API: the hosted catalogue rejects nvext
    outright (400, 'unknown field guided_json') — that extension belongs to
    self-hosted NIM containers. Regressing this silently returns prose."""
    prompt, extra = _client(monkeypatch, "nvidia")._structured("Rate this.", SCHEMA)
    assert extra["response_format"]["type"] == "json_schema"
    assert "nvext" not in extra
    assert prompt == "Rate this."


def test_nvext_remains_available_for_self_hosted_nim(monkeypatch, tmp_path):
    """Still the right dialect for a NIM container someone runs themselves,
    reachable through the custom provider."""
    monkeypatch.setenv("PUBLIKCLIP_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("PUBLIKCLIP_LLM_MODEL", "local-nim")
    monkeypatch.setenv("PUBLIKCLIP_LLM_STRUCTURED", "nvext")
    prompt, extra = llm.make_client("custom")._structured("Rate this.", SCHEMA)
    assert extra == {"nvext": {"guided_json": SCHEMA}}
    assert prompt == "Rate this."


def test_json_schema_uses_the_openai_response_format(monkeypatch):
    prompt, extra = _client(monkeypatch, "openai")._structured("Rate this.", SCHEMA)
    assert extra["response_format"]["type"] == "json_schema"
    assert extra["response_format"]["json_schema"]["schema"] == SCHEMA
    assert prompt == "Rate this."


def test_json_object_puts_the_schema_in_the_prompt(monkeypatch):
    """json_object guarantees valid JSON, not the right shape — the schema
    has to reach the model some other way."""
    prompt, extra = _client(monkeypatch, "groq")._structured("Rate this.", SCHEMA)
    assert extra == {"response_format": {"type": "json_object"}}
    assert "hook" in prompt and prompt != "Rate this."


# --- vision routing -------------------------------------------------------


class _Recorder:
    """Stands in for httpx.post and remembers what it was handed."""

    def __init__(self, payload):
        self.payload = payload
        self.bodies = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.bodies.append(json)
        return _Response(self.payload)


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def test_images_route_to_the_vision_model_text_to_the_text_model(monkeypatch):
    client = _client(monkeypatch, "nvidia")
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)

    client.generate_json("text only", SCHEMA)
    client.generate_json("with frames", SCHEMA, images=[b"\xff\xd8\xffjpeg"])

    assert post.bodies[0]["model"] == client.model
    assert post.bodies[1]["model"] == client.vision_model
    assert isinstance(post.bodies[0]["messages"][0]["content"], str)
    parts = post.bodies[1]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_images_are_dropped_when_the_provider_cannot_see(monkeypatch):
    client = _client(monkeypatch, "groq")
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)

    client.generate_json("with frames", SCHEMA, images=[b"\xff\xd8\xffjpeg"])

    assert post.bodies[0]["model"] == client.model
    assert isinstance(post.bodies[0]["messages"][0]["content"], str)


def test_text_and_vision_calls_do_not_share_a_cache_entry(monkeypatch):
    """Same prompt, different model — a T2 call must not be served a T1
    answer just because the text matched."""
    client = _client(monkeypatch, "nvidia")
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)

    client.generate_json("same words", SCHEMA)
    client.generate_json("same words", SCHEMA, images=[b"\xff\xd8\xffjpeg"])
    assert len(post.bodies) == 2

    # ...and a genuine repeat is served from cache, not re-spent.
    client.generate_json("same words", SCHEMA)
    assert len(post.bodies) == 2


def test_model_override_applies_to_a_preset(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_LLM_MODEL", "meta/llama-3.3-70b-instruct")
    client = _client(monkeypatch, "nvidia")
    assert client.model == "meta/llama-3.3-70b-instruct"


# --- errors ---------------------------------------------------------------


class _ErrorResponse(_Response):
    def __init__(self, status_code, payload):
        super().__init__({})
        self.status_code = status_code
        self._error = payload

    def json(self):
        return self._error


def test_rejected_key_says_so_instead_of_retrying(monkeypatch):
    client = _client(monkeypatch, "nvidia")
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return _ErrorResponse(401, {"error": {"message": "invalid key"}})

    monkeypatch.setattr(llm.httpx, "post", post)
    with pytest.raises(llm.LlmError) as err:
        client.generate_json("hi", SCHEMA)
    assert "rejected the API key" in str(err.value)
    assert len(calls) == 1, "an auth failure must not burn retries"


def test_unknown_model_points_at_the_override(monkeypatch):
    client = _client(monkeypatch, "nvidia")
    monkeypatch.setattr(
        llm.httpx, "post", lambda *a, **k: _ErrorResponse(404, {"error": "no such model"})
    )
    with pytest.raises(llm.LlmError) as err:
        client.generate_json("hi", SCHEMA)
    assert "PUBLIKCLIP_LLM_MODEL" in str(err.value)


def test_rate_limit_surfaces_the_servers_own_words(monkeypatch):
    client = _client(monkeypatch, "nvidia")
    monkeypatch.setattr(
        llm.httpx,
        "post",
        lambda *a, **k: _ErrorResponse(429, {"error": {"message": "credits exhausted"}}),
    )
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda s: None)  # don't actually back off
    with pytest.raises(llm.LlmError) as err:
        client.generate_json("hi", SCHEMA)
    assert "credits exhausted" in str(err.value)


# --- reasoning models ------------------------------------------------------


def test_openrouter_defaults_to_ox_alpha_with_a_seeing_companion(monkeypatch):
    """The text judge and the frame reader are different models on purpose."""
    client = _client(monkeypatch, "openrouter")
    assert client.model == "stealth/ox-alpha"
    assert client.vision_model and client.vision_model != client.model


def test_reasoning_effort_is_sent_when_the_provider_declares_it(monkeypatch):
    client = _client(monkeypatch, "openrouter")
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)
    client.generate_json("rate this", SCHEMA)
    assert post.bodies[0]["reasoning"] == {"effort": "low"}


def test_a_provider_without_reasoning_sends_none(monkeypatch):
    """Sending the parameter to a server that does not know it is noise."""
    client = _client(monkeypatch, "nvidia")
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)
    client.generate_json("rate this", SCHEMA)
    assert "reasoning" not in post.bodies[0]


def test_reasoning_effort_is_overridable(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_LLM_REASONING", "high")
    client = _client(monkeypatch, "openrouter")
    assert client.reasoning == "high"


def test_reasoning_can_be_turned_off(monkeypatch):
    """Thinking costs tokens on every one of ~60 calls per source."""
    monkeypatch.setenv("PUBLIKCLIP_LLM_REASONING", "off")
    client = _client(monkeypatch, "openrouter")
    assert client.reasoning is None
    post = _Recorder({"hook": 7})
    monkeypatch.setattr(llm.httpx, "post", post)
    client.generate_json("rate this", SCHEMA)
    assert "reasoning" not in post.bodies[0]


def test_reasoning_can_be_enabled_on_a_provider_that_has_none(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_LLM_REASONING", "medium")
    client = _client(monkeypatch, "nvidia")
    assert client.reasoning == "medium"
