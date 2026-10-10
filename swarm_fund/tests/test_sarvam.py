"""Sarvam AI provider, against faked HTTP replies (no key or network needed)."""
import json

import pytest
import requests

import agents
import config
from tests.test_agents import briefs

SCHEMA = agents.SCHEMAS["analyst"]


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        return self._payload


def reply(content, finish="stop", refusal=None):
    return FakeResponse(200, {"choices": [{"index": 0, "finish_reason": finish,
                                           "message": {"role": "assistant", "content": content,
                                                       "refusal": refusal}}]})


@pytest.fixture
def calls(monkeypatch):
    """Queue of responses for requests.post; records every request."""
    state = {"queue": [], "sent": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["sent"].append({"url": url, "body": json, "headers": headers, "timeout": timeout})
        item = state["queue"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(agents.time, "sleep", lambda s: None)
    return state


def llm():
    return agents.SarvamLLM(api_key="test-key")


def test_request_shape(calls):
    calls["queue"].append(reply('{"score": 1, "summary": "s", "points": []}'))
    obj, err = llm().json_call("sarvam-105b", "SYSTEM", "USER", SCHEMA, 4000, "low")
    assert err is None and obj["score"] == 1
    sent = calls["sent"][0]
    assert sent["url"] == "https://api.sarvam.ai/v1/chat/completions"
    assert sent["headers"]["api-subscription-key"] == "test-key"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    b = sent["body"]
    assert b["model"] == "sarvam-105b" and b["max_tokens"] == 4000 and b["temperature"] == 0.2
    assert b["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert b["response_format"] == {"type": "json_schema",
                                    "json_schema": {"name": "reply", "schema": SCHEMA, "strict": True}}
    assert b["reasoning_effort"] == "low"


def test_medium_effort_is_left_to_sarvams_default(calls):
    calls["queue"].append(reply('{"score": 0, "summary": "s", "points": []}'))
    llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "medium")
    assert "reasoning_effort" not in calls["sent"][0]["body"]


def test_think_block_is_stripped(calls):
    calls["queue"].append(reply('<think>score {"score": -2} maybe</think>\n{"score": 2, "summary": "x", "points": []}'))
    obj, err = llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "low")
    assert obj["score"] == 2


def test_schema_rejection_falls_back_to_json_mode(calls):
    calls["queue"] += [FakeResponse(400, {"error": "unsupported response_format"}),
                       reply('{"score": -1, "summary": "s", "points": []}')]
    obj, err = llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "low")
    assert obj["score"] == -1
    second = calls["sent"][1]["body"]
    assert second["response_format"] == {"type": "json_object"}
    assert second["messages"][0]["content"].endswith("Reply with a single JSON object only.")


@pytest.mark.parametrize("first", [FakeResponse(429, {"error": "rate"}, {"Retry-After": "2"}),
                                   FakeResponse(503, {"error": "model_overloaded"}),
                                   requests.Timeout("slow")])
def test_one_retry_on_rate_limit_overload_or_timeout(calls, first):
    calls["queue"] += [first, reply('{"score": 1, "summary": "s", "points": []}')]
    obj, err = llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "low")
    assert obj["score"] == 1 and len(calls["sent"]) == 2


def test_second_failure_gives_up_without_raising(calls):
    calls["queue"] += [FakeResponse(503, {"error": "x"}), FakeResponse(503, {"error": "x"})]
    obj, err = llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "low")
    assert obj is None and "503" in err and len(calls["sent"]) == 2


@pytest.mark.parametrize("resp, message", [
    (FakeResponse(403, {"error": "invalid_api_key_error"}), "rejected the API key"),
    (FakeResponse(404, {"error": "model"}), "does not know model"),
    (reply('{"score": 1', finish="length"), "cut off at max_tokens"),
    (reply(None, refusal="cannot help"), "declined"),
    (reply("", finish="content_filter"), "content filter"),
    (reply("   "), "empty reply"),
])
def test_failures_are_reported_not_raised(calls, resp, message):
    calls["queue"].append(resp)
    obj, err = llm().json_call("sarvam-105b", "S", "U", SCHEMA, 4000, "low")
    assert obj is None and message in err
    assert len(calls["sent"]) == 1                    # no pointless retry


def test_full_swarm_on_sarvam(calls):
    analyst = '{"score": 1, "summary": "ok", "points": ["close 100 above 200-day 90"]}'
    debate = '{"case": "c", "arguments": ["a"]}'
    manager = json.dumps({"action": "BUY", "confidence": 71, "thesis": ["a 1", "b 2", "c 3"],
                          "kill_condition": "close below 95"})
    calls["queue"] += [reply(analyst)] * 4 + [reply(debate)] * 2 + [reply(manager)]
    res = agents.run_swarm("AAA", "IT", False, briefs(), llm())
    assert res["agent_mode"] == "llm:sarvam" and res["action"] == "BUY" and res["confidence"] == 71
    assert len(calls["sent"]) == 7
    assert all(r["model"] in (config.ANALYST_MODEL, config.MANAGER_MODEL) for r in res["io"])
    assert all("fallbacks" not in c["body"] for c in calls["sent"])     # Anthropic-only option


def test_make_llm_selects_provider(monkeypatch):
    monkeypatch.setattr(config, "SARVAM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_PROVIDER", "sarvam")
    assert isinstance(agents.make_llm(), agents.SarvamLLM) and agents.make_llm().enabled
    assert not agents.make_llm(offline=True).enabled                    # offline never calls a model
    assert type(agents.make_llm(provider="anthropic")) is agents.LLM
    assert agents.key_name("sarvam") == "SARVAM_API_KEY"
    with pytest.raises(ValueError, match="deepseek"):
        agents.make_llm(provider="deepseek")


def test_unknown_provider_aborts_the_run(monkeypatch, tmp_path):
    import pipeline
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    with pytest.raises(pipeline.RunError, match="not one of"):
        pipeline.run(str(tmp_path / "x.db"), offline=False, asof="2026-10-09")
