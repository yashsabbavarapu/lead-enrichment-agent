"""CLI surface tests: help text, `doctor`, and `doctor --models`.

No network and no real LLM — `list_models` is mocked and the environment is
scrubbed per test, because `doctor`'s whole job is reporting the environment.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lea import cli as cli_mod
from lea.config import OPENAI_COMPAT_PRESETS

runner = CliRunner()

# Every env var that can steer provider resolution.
_PROVIDER_ENV = ["ANTHROPIC_API_KEY", "LEA_LLM_PROVIDER", "LEA_MODEL", "LEA_LLM_BASE_URL",
                 "LEA_LLM_API_KEY"] + [p["env"] for p in OPENAI_COMPAT_PRESETS.values() if p["env"]]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """A keyless machine, and no .env bleeding in from the repo."""
    for name in _PROVIDER_ENV + ["TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY", "LEA_OFFLINE"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: 0)


def run(*args):
    return runner.invoke(cli_mod.app, list(args))


def flat(text: str) -> str:
    """Collapse rich's wrapping so assertions are about words, not line breaks."""
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Help text
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["enrich", "batch"])
def test_provider_help_lists_every_supported_provider(command):
    """A stale help string is how someone concludes a feature does not exist."""
    out = run(command, "--help").output
    for provider in ("anthropic", "google", "groq", "openrouter", "cerebras", "ollama", "stub"):
        assert provider in out, f"{command} --help does not mention {provider}"


def test_icp_infer_help_lists_providers():
    out = run("icp", "infer", "--help").output
    for provider in ("google", "groq", "openrouter", "cerebras", "ollama"):
        assert provider in out


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_keyless_leads_with_the_free_options():
    result = run("doctor")
    assert result.exit_code == 0
    out = result.output

    # every provider is offered, by its preset label
    for preset in OPENAI_COMPAT_PRESETS.values():
        assert preset["label"] in out
        if preset["env"]:
            assert preset["env"] in out
    assert "ANTHROPIC_API_KEY" in out

    # the free ones are visibly free, and the paid one visibly paid
    assert out.count("FREE") >= 5
    assert "paid" in out

    # the stub is what is actually running, and says so
    assert "stub" in out
    assert "Ready, keyless" in out
    # and the user is told exactly where to get a free key
    assert "aistudio.google.com" in out
    assert "console.groq.com" in out


def test_doctor_never_prints_key_values(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-super-secret-value-123")
    out = run("doctor").output
    assert "sk-super-secret-value-123" not in out
    assert "present" in out


@pytest.mark.parametrize("provider", list(OPENAI_COMPAT_PRESETS))
def test_doctor_resolves_and_reports_each_provider(provider, monkeypatch):
    preset = OPENAI_COMPAT_PRESETS[provider]
    if preset["env"]:
        monkeypatch.setenv(preset["env"], "test-key")
    else:  # ollama has no key, so it needs selecting explicitly
        monkeypatch.setenv("LEA_LLM_PROVIDER", provider)

    result = run("doctor")
    assert result.exit_code == 0
    out = result.output

    assert f"> {preset['label'][:25]}" in out, "the active provider must be marked"
    # the resolved endpoint and model are shown so a user can see what will be called
    assert preset["base_url"].rstrip("/")[:40] in out.replace("\n", "")
    assert preset["model"] in out.replace("\n", "")
    assert "Ready." in out


def test_doctor_marks_anthropic_when_its_key_is_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    out = run("doctor").output
    assert "> Anthropic (Claude)" in out
    assert "Ready." in out
    assert "sk-ant-xxx" not in out


# --------------------------------------------------------------------------
# doctor --models
# --------------------------------------------------------------------------


def _mock_models(monkeypatch, models, exc=None):
    import lea.llm.factory as factory

    def fake(settings, provider=None):
        if exc:
            raise exc
        return list(models)

    monkeypatch.setattr(factory, "list_models", fake)


def test_doctor_models_lists_what_the_provider_serves(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    served = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b"]
    _mock_models(monkeypatch, served)

    result = run("doctor", "--models")
    assert result.exit_code == 0
    out = flat(result.output)
    for m in served:
        assert m in out
    # the preset default is in this list, so it is flagged as selected, not as broken
    assert "currently selected" in out
    assert "does not list" not in out


def test_doctor_models_warns_when_the_configured_id_is_missing(monkeypatch):
    """The stale-model case: the endpoint answers, but not with the id we default to."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, ["llama-3.1-8b-instant", "gemma2-9b-it"])

    out = flat(run("doctor", "--models").output)
    assert "does not list" in out
    assert "404" in out
    assert "LEA_MODEL" in out


def test_doctor_models_explains_an_empty_list(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, [])

    out = flat(run("doctor", "--models").output)
    assert "no models returned" in out
    assert "unreachable" in out or "rejected" in out
    assert "LEA_MODEL" in out


def test_doctor_models_survives_a_raising_lookup(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, [], exc=RuntimeError("boom"))

    result = run("doctor", "--models")
    assert result.exit_code == 0
    assert "boom" in result.output or "no models returned" in result.output


def test_doctor_models_says_why_for_stub_and_anthropic(monkeypatch):
    out = flat(run("doctor", "--models").output)
    assert "does not expose a model list" in out

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")
    out = flat(run("doctor", "--models").output)
    assert "does not expose a model list" in out


def test_doctor_models_does_not_hang_on_a_stuck_endpoint(monkeypatch):
    """A wedged endpoint must not wedge doctor."""
    import lea.llm.factory as factory

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(cli_mod, "MODELS_TIMEOUT_S", 0.2)

    def never_returns(settings, provider=None):
        import time

        time.sleep(30)
        return []

    monkeypatch.setattr(factory, "list_models", never_returns)

    result = run("doctor", "--models")
    assert result.exit_code == 0
    assert "took longer than" in flat(result.output)


def test_doctor_without_models_flag_makes_no_lookup(monkeypatch):
    import lea.llm.factory as factory

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    called = []
    monkeypatch.setattr(factory, "list_models", lambda *a, **k: called.append(1) or [])

    run("doctor")
    assert not called, "doctor must not hit the network unless --models is passed"
