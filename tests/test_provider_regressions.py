"""Provider contracts tested with SDK and HTTP doubles, without network access."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexuseval import CacheConfig, CacheManager, LLMClient, LLMConfig
from nexuseval.providers import LLMProviderRegistry
from nexuseval.providers.base import LLMMessage


@pytest.fixture(params=["openai", "anthropic", "google", "groq", "ollama"])
def provider_harness(request, monkeypatch):
    name = request.param
    module = importlib.import_module(f"nexuseval.providers.{name}_provider")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, input_tokens=10, output_tokens=5)
    message = SimpleNamespace(content='{"score": 0.8, "reason": "OK"}')
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], content=[SimpleNamespace(text=message.content)],
        usage=usage, text=message.content,
    )
    api_request = AsyncMock(return_value=response)
    client = MagicMock()
    client.chat.completions.create = api_request
    client.messages.create = api_request
    client.generate_content_async = api_request
    client.start_chat.return_value.send_message_async = api_request
    flag = {"anthropic": "ANTHROPIC_AVAILABLE", "google": "GOOGLE_AVAILABLE",
            "groq": "GROQ_AVAILABLE", "ollama": "AIOHTTP_AVAILABLE"}.get(name)
    if flag:
        monkeypatch.setattr(module, flag, True)
    if name == "google":
        monkeypatch.setattr(module, "genai", SimpleNamespace(
            configure=MagicMock(), GenerativeModel=MagicMock(return_value=client),
            types=SimpleNamespace(GenerationConfig=lambda **kwargs: kwargs),
        ), raising=False)
    elif name == "ollama":
        http_response = MagicMock()
        http_response.json = api_request
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=http_response)
        session = MagicMock()
        session.post.return_value = context
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        monkeypatch.setattr(module, "aiohttp", SimpleNamespace(
            ClientSession=MagicMock(return_value=session_context),
            ClientTimeout=lambda **kwargs: kwargs,
        ), raising=False)
    else:
        constructor_name = {"openai": "AsyncOpenAI", "anthropic": "AsyncAnthropic", "groq": "AsyncGroq"}[name]
        constructor = MagicMock(return_value=client)
        monkeypatch.setattr(module, constructor_name, constructor, raising=False)
    provider_class = getattr(module, {"openai": "OpenAI", "anthropic": "Anthropic",
                                     "google": "Google", "groq": "Groq", "ollama": "Ollama"}[name] + "Provider")
    provider = provider_class(
        model="test-model", api_key="test", temperature=0.6, max_tokens=42,
        max_retries=0, retry_delay=0, timeout=9,
    )

    def set_content(content):
        message.content = content
        response.content[0].text = content
        response.text = content
        if name == "ollama":
            api_request.return_value = {
                "response": content, "message": {"content": content},
                "prompt_eval_count": 10, "eval_count": 5,
            }

    set_content(message.content)
    return SimpleNamespace(name=name, provider=provider, request=api_request, client=client,
                           set_content=set_content, module=module,
                           http_response=http_response if name == "ollama" else None,
                           session=session if name == "ollama" else None)


@pytest.mark.asyncio
async def test_zero_retries_still_makes_one_attempt(provider_harness):
    harness = provider_harness
    result = await harness.provider.generate_json("Return JSON")
    assert result == {"score": 0.8, "reason": "OK"}
    assert harness.request.await_count == 1
    assert harness.provider.get_usage_stats()["total_tokens"] > 0


@pytest.mark.asyncio
async def test_failed_json_raises_and_usage_counts_all_attempts(provider_harness):
    harness = provider_harness
    harness.provider.max_retries = 2
    harness.set_content("invalid JSON")
    with pytest.raises(ValueError):
        await harness.provider.generate_json("Return JSON")
    assert harness.request.await_count == 3
    per_attempt = (
        harness.provider.count_tokens("Return JSON") + harness.provider.count_tokens("invalid JSON")
        if harness.name == "google" else 15
    )
    assert harness.provider.get_usage_stats()["total_tokens"] == 3 * per_attempt


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["[]", "null", '"text"'])
async def test_json_must_be_an_object(provider_harness, content):
    harness = provider_harness
    harness.set_content(content)
    with pytest.raises(ValueError, match="JSON object"):
        await harness.provider.generate_json("Return JSON")


@pytest.mark.asyncio
async def test_provider_errors_propagate_instead_of_returning_scores(provider_harness):
    harness = provider_harness
    harness.request.side_effect = RuntimeError("Service unavailable")
    with pytest.raises(RuntimeError, match="Service unavailable"):
        await harness.provider.generate_json("Return JSON")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "generate_json", "chat"])
async def test_generation_defaults_and_per_call_overrides(provider_harness, operation):
    harness = provider_harness
    method = getattr(harness.provider, operation)
    argument = [LLMMessage(role="user", content="Q")] if operation == "chat" else "Q"
    for overrides, expected_temperature, expected_tokens in (({}, 0.6, 42), ({"temperature": 0.2, "max_tokens": 7}, 0.2, 7)):
        await method(argument, **overrides)
        if harness.name == "ollama":
            options = harness.session.post.call_args.kwargs["json"]["options"]
            assert options == {"temperature": expected_temperature, "num_predict": expected_tokens}
        else:
            api = harness.client.start_chat.return_value.send_message_async if operation == "chat" and harness.name == "google" else harness.request
            options = api.call_args.kwargs
            if harness.name == "google":
                options = options["generation_config"]
                assert options["max_output_tokens"] == expected_tokens
            else:
                assert options["max_tokens"] == expected_tokens
            assert options["temperature"] == expected_temperature


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_harness", ["ollama"], indirect=True)
async def test_ollama_checks_http_status(provider_harness):
    harness = provider_harness
    harness.http_response.raise_for_status.side_effect = RuntimeError("HTTP 500")
    for operation, argument in (("generate", "Q"), ("generate_json", "Q"), ("chat", [LLMMessage(role="user", content="Q")])):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await getattr(harness.provider, operation)(argument)
    assert harness.request.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_harness", ["google"], indirect=True)
async def test_google_chat_preserves_system_and_all_user_messages(provider_harness):
    harness = provider_harness
    messages = [LLMMessage(role="system", content="Be concise"),
                LLMMessage(role="user", content="First"), LLMMessage(role="user", content="Second")]
    await harness.provider.chat(messages)
    assert harness.client.start_chat.call_args.kwargs["history"] == [
        {"role": "user", "parts": ["Be concise\n\nFirst"]}
    ]
    assert harness.client.start_chat.return_value.send_message_async.call_args.args == ("Second",)
    with pytest.raises(ValueError, match="final user"):
        await harness.provider.chat([])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_harness", ["anthropic"], indirect=True)
async def test_anthropic_chat_omits_absent_system_parameter(provider_harness):
    harness = provider_harness
    await harness.provider.chat([LLMMessage(role="user", content="Q")])
    assert "system" not in harness.request.call_args.kwargs


@pytest.mark.asyncio
async def test_client_cache_separates_generation_defaults_and_endpoints(monkeypatch):
    provider = AsyncMock()
    provider.generate_json.return_value = {"score": 0.8, "reason": "OK"}
    monkeypatch.setattr(LLMProviderRegistry, "create", lambda **kwargs: provider)
    cache = CacheManager.from_config(CacheConfig())
    clients = [LLMClient(config=LLMConfig(temperature=temperature, base_url=url), cache_manager=cache)
               for temperature, url in [(0.1, "http://first"), (0.9, "http://first"), (0.1, "http://second")]]
    for client in clients:
        await client.get_score("same prompt")
        await client.get_score("same prompt")
    assert provider.generate_json.await_count == 3


@pytest.mark.asyncio
async def test_client_does_not_cache_provider_failure(monkeypatch):
    provider = AsyncMock()
    provider.generate_json.side_effect = [RuntimeError("offline"), {"score": 1.0, "reason": "OK"}]
    monkeypatch.setattr(LLMProviderRegistry, "create", lambda **kwargs: provider)
    client = LLMClient()
    with pytest.raises(RuntimeError, match="offline"):
        await client.get_score("Q")
    assert await client.get_score("Q") == {"score": 1.0, "reason": "OK"}
    assert provider.generate_json.await_count == 2


def test_optional_provider_availability_matches_installed_sdk(monkeypatch):
    registry = importlib.import_module("nexuseval.providers.registry")
    for name, flag in [("anthropic", "ANTHROPIC_AVAILABLE"), ("google", "GOOGLE_AVAILABLE"),
                       ("groq", "GROQ_AVAILABLE"), ("ollama", "AIOHTTP_AVAILABLE")]:
        module = importlib.import_module(f"nexuseval.providers.{name}_provider")
        monkeypatch.setattr(module, flag, False)
        monkeypatch.setattr(registry, "_" + ("OLLAMA" if name == "ollama" else name.upper()) + "_AVAILABLE", False)
        monkeypatch.delitem(registry._PROVIDER_CLASSES, name, raising=False)
        assert LLMProviderRegistry.list_providers()[name] is False
        with pytest.raises(ImportError, match="not installed"):
            LLMProviderRegistry.get(name)


@pytest.mark.asyncio
async def test_schema_validation_happens_before_caching(monkeypatch):
    from pydantic import ValidationError
    from nexuseval.metrics.advanced import BiasResponse
    provider = AsyncMock()
    provider.generate_json.side_effect = [{}, {"has_bias": False, "severity": 0.0}]
    monkeypatch.setattr(LLMProviderRegistry, "create", lambda **kwargs: provider)
    client = LLMClient()
    with pytest.raises(ValidationError):
        await client.get_score("Q", response_model=BiasResponse)
    assert await client.get_score("Q", response_model=BiasResponse) == {"has_bias": False, "severity": 0.0}
    assert provider.generate_json.await_count == 2


@pytest.mark.asyncio
async def test_client_cache_separates_response_schemas(monkeypatch):
    from nexuseval.metrics.standard import ScoreResponse

    class StricterScoreResponse(ScoreResponse):
        evidence: str

    provider = AsyncMock()
    provider.generate_json.side_effect = [
        {}, {"score": 1.0, "reason": "OK"},
        {"score": 1.0, "reason": "OK", "evidence": "Source"},
    ]
    monkeypatch.setattr(LLMProviderRegistry, "create", lambda **kwargs: provider)
    client = LLMClient()
    assert await client.get_score("same prompt") == {}
    for schema in (ScoreResponse, StricterScoreResponse):
        first = await client.get_score("same prompt", response_model=schema)
        assert await client.get_score("same prompt", response_model=schema) == first
    assert provider.generate_json.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "chat"])
async def test_text_and_chat_requests_use_provider_retry_settings(provider_harness, operation):
    harness = provider_harness
    harness.provider.max_retries = 2
    good_response = harness.request.return_value
    harness.request.side_effect = [RuntimeError("offline"), good_response]
    argument = [LLMMessage(role="user", content="Q")] if operation == "chat" else "Q"
    await getattr(harness.provider, operation)(argument)
    assert harness.request.await_count == 2
