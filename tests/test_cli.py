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
    # _MODEL_RESOLUTION is process-global and would leak between tests
    from lea.llm.openai_compat import reset_provider_state

    reset_provider_state()
    yield
    reset_provider_state()


def run(*args):
    return runner.invoke(cli_mod.app, list(args))


def _boom(*a, **k):
    raise AssertionError("no network lookup should happen here")


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
    """Patch at openai_compat: factory.list_models delegates to it, and
    resolve_model calls it directly, so one patch covers both paths (and keeps
    resolve_model off the network)."""
    import lea.llm.openai_compat as oc

    def fake(settings, provider=None):
        if exc:
            raise exc
        return list(models)

    monkeypatch.setattr(oc, "list_models", fake)


def test_doctor_models_outcome_served(monkeypatch):
    """Outcome 1: the configured id really is served — say so plainly."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    served = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b"]
    _mock_models(monkeypatch, served)

    result = run("doctor", "--models")
    assert result.exit_code == 0
    out = flat(result.output)

    for m in served:
        assert m in out
    assert "llama-3.3-70b-versatile — served, will be used." in out
    assert "will be used" in out
    # the stale-id language must NOT appear for a healthy setup
    assert "is not served here" not in out
    assert "runs will fail" not in out


def test_doctor_models_outcome_substituted(monkeypatch):
    """Outcome 2: configured id is gone, but the run self-heals — report the
    replacement, not the old 'runs will fail with a 404' warning."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, ["llama-3.1-8b-instant", "gemma2-9b-it"])

    out = flat(run("doctor", "--models").output)

    assert "llama-3.3-70b-versatile is not served here" in out
    assert "will be used automatically" in out
    assert "Nothing to do — runs self-heal" in out
    assert "LEA_MODEL to pin a different one" in out
    # the obsolete scare line is gone
    assert "Runs will likely fail with a 404" not in out


def test_doctor_models_outcome_unresolvable(monkeypatch):
    """Outcome 3: nothing usable — the hard warning is still correct."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, [])

    out = flat(run("doctor", "--models").output)
    assert "no models returned" in out
    assert "unreachable" in out or "rejected" in out
    assert "no replacement could be found" in out
    assert "runs will fail" in out
    assert "LEA_MODEL" in out


def test_doctor_models_warms_the_cache_for_the_next_run(monkeypatch):
    """resolve_model shares its cache with the agent: doctor + a run should pay
    one lookup between them."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    _mock_models(monkeypatch, ["llama-3.1-8b-instant"])

    run("doctor", "--models")

    from lea.config import Settings

    settings = Settings.from_env()
    assert cli_mod.cached_resolved_model(settings, "groq") == "llama-3.1-8b-instant"

    # ...and a following bare `doctor` reports it without any lookup
    import lea.llm.openai_compat as oc

    monkeypatch.setattr(oc, "list_models", _boom)
    out = flat(run("doctor").output)
    assert "will actually use llama-3.1-8b-instant" in out
    assert "does not serve llama-3.3-70b-versatile" in out


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
    """A cold cache must not become a network call. doctor is what people run when
    something is already broken, so it has to be instant and work offline."""
    import lea.llm.factory as factory
    import lea.llm.openai_compat as oc

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(factory, "list_models", _boom)
    monkeypatch.setattr(oc, "list_models", _boom)
    monkeypatch.setattr(oc, "resolve_model", _boom)

    result = run("doctor")
    assert result.exit_code == 0
    out = flat(result.output)
    # falls back to reporting the configured id, and points at --models
    assert "model=llama-3.3-70b-versatile" in out
    assert "lea doctor --models" in out


def test_cached_resolved_model_is_none_when_cold(monkeypatch):
    from lea.config import Settings

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert cli_mod.cached_resolved_model(Settings.from_env(), "groq") is None


def test_cached_resolved_model_survives_a_moved_internal(monkeypatch):
    """The cache peek reaches into openai_compat internals on purpose; if that name
    moves, doctor must degrade to 'unknown', never crash."""
    import lea.llm.openai_compat as oc
    from lea.config import Settings

    class Moved(dict):
        def get(self, *a, **k):  # the shape doctor depends on, gone
            raise AttributeError("_MODEL_RESOLUTION moved")

    monkeypatch.setattr(oc, "_MODEL_RESOLUTION", Moved())
    assert cli_mod.cached_resolved_model(Settings.from_env(), "groq") is None
    assert run("doctor").exit_code == 0


# --------------------------------------------------------------------------
# --offline does not mean the LLM is offline
# --------------------------------------------------------------------------


def test_offline_warns_that_the_llm_is_still_live(monkeypatch):
    """One run is ~11 calls against a ~20/day free tier, so a silent burn during
    'offline' iteration costs a user their whole day's quota."""
    from lea.config import Settings

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = Settings.from_env()
    settings.offline = True

    err: list[str] = []
    monkeypatch.setattr(cli_mod.err_console, "print", lambda m, *a, **k: err.append(str(m)))
    cli_mod.warn_if_offline_but_live_llm(settings)

    assert err, "an offline run against a live provider must say so"
    joined = flat(" ".join(err))
    assert "groq" in joined
    assert "--provider stub" in joined
    assert "quota" in joined


def test_no_warning_when_the_run_really_is_offline(monkeypatch):
    from lea.config import Settings

    err: list[str] = []
    monkeypatch.setattr(cli_mod.err_console, "print", lambda m, *a, **k: err.append(str(m)))

    stub = Settings.from_env()          # keyless -> stub
    stub.offline = True
    cli_mod.warn_if_offline_but_live_llm(stub)
    assert not err, "stub + offline is genuinely offline"

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    online = Settings.from_env()        # live provider but NOT offline: no surprise
    online.offline = False
    cli_mod.warn_if_offline_but_live_llm(online)
    assert not err


def test_offline_help_text_is_honest():
    out = flat(run("enrich", "--help").output)
    assert "LLM still makes live calls" in out
    assert "--provider stub" in out


def test_doctor_offline_row_is_honest(monkeypatch):
    monkeypatch.setenv("LEA_OFFLINE", "1")
    out = flat(run("doctor").output)
    assert "the LLM is still live unless provider=stub" in out
