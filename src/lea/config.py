"""Runtime settings. Everything is overridable by env or CLI flag.

Design rule: the tool must run end-to-end with zero API keys. Keys upgrade it,
they are never required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent.parent
FIXTURE_DIR = PROJECT_ROOT / "examples" / "fixtures"

DEFAULT_MODEL = "claude-sonnet-5"
PROJECT_URL = "https://github.com/your-handle/lead-enrichment-agent"  # PLACEHOLDER: your repo


def build_user_agent(contact: Optional[str] = None) -> str:
    """Identify the crawler honestly.

    Politeness matters here: these requests land in other people's logs, and a
    contactable UA is the difference between a research bot and an anonymous
    scraper. Set LEA_CONTACT to an email or URL you actually read.
    """
    who = contact or os.environ.get("LEA_CONTACT") or ""
    tail = f"; contact: {who}" if who else ""
    return f"lead-enrichment-agent/0.1 (+{PROJECT_URL}{tail})"


USER_AGENT = build_user_agent()


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v else default


def _env_flag(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() not in ("", "0", "false", "no")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default



# OpenAI-compatible providers. One adapter serves all of them: they differ only
# in base URL, auth env var and model naming. Free tiers exist for google, groq,
# openrouter and cerebras; ollama is local and needs no key at all.
#
# Default models are a starting point, not a guarantee — free lineups rotate.
# `lea doctor --models` lists what the configured provider will actually serve.
OPENAI_COMPAT_PRESETS: dict[str, dict[str, str]] = {
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env": "GOOGLE_API_KEY",
        "model": "gemini-2.0-flash",
        "label": "Google AI Studio (Gemini)",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "label": "Groq",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "label": "OpenRouter",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env": "CEREBRAS_API_KEY",
        "model": "llama-3.3-70b",
        "label": "Cerebras",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env": "",  # local, no key
        "model": "qwen2.5:7b",
        "label": "Ollama (local)",
    },
}

# Tried in order when llm_provider is "auto".
_AUTO_LLM_ORDER = ("anthropic", "google", "groq", "openrouter", "cerebras")


@dataclass
class Settings:
    # --- providers -------------------------------------------------------
    # auto | anthropic | google | groq | openrouter | cerebras | ollama | stub
    llm_provider: str = "auto"
    model: str = DEFAULT_MODEL
    anthropic_api_key: Optional[str] = None
    # Any OpenAI-compatible endpoint. Set explicitly to override a preset,
    # or to point at a provider this build has never heard of.
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    search_provider: str = "auto"  # auto | tavily | serper | brave | duckduckgo | fixture
    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None

    # --- agent budget ----------------------------------------------------
    max_steps: int = 12
    max_searches: int = 5
    max_fetches: int = 6
    step_timeout_s: float = 20.0
    total_timeout_s: float = 180.0

    # --- network ---------------------------------------------------------
    http_timeout_s: float = 15.0
    max_page_chars: int = 12_000
    max_snippet_chars: int = 1_200
    user_agent: str = field(default_factory=build_user_agent)
    respect_robots: bool = True

    # --- offline / demo --------------------------------------------------
    offline: bool = False          # serve every HTTP call from examples/fixtures
    record_fixtures: bool = False  # write live responses into examples/fixtures
    fixture_dir: Path = field(default_factory=lambda: FIXTURE_DIR)

    # --- output ----------------------------------------------------------
    out_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "out")
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=_env("LEA_LLM_PROVIDER", "auto"),
            model=_env("LEA_MODEL", DEFAULT_MODEL),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_base_url=_env("LEA_LLM_BASE_URL"),
            llm_api_key=_env("LEA_LLM_API_KEY"),
            search_provider=_env("LEA_SEARCH_PROVIDER", "auto"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            serper_api_key=_env("SERPER_API_KEY"),
            brave_api_key=_env("BRAVE_API_KEY"),
            max_steps=_env_int("LEA_MAX_STEPS", 12),
            max_searches=_env_int("LEA_MAX_SEARCHES", 5),
            max_fetches=_env_int("LEA_MAX_FETCHES", 6),
            offline=_env_flag("LEA_OFFLINE"),
            record_fixtures=_env_flag("LEA_RECORD_FIXTURES"),
            user_agent=build_user_agent(),
        )

    # --- resolution helpers ---------------------------------------------
    def resolved_llm(self) -> str:
        """Which brain will actually run.

        A key is never required: every path falls through to the deterministic
        stub rather than erroring, so a keyless clone still works.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        for candidate in _AUTO_LLM_ORDER:
            if self.key_for(candidate):
                return candidate
        return "stub"

    def key_for(self, provider: str) -> Optional[str]:
        """The API key for *provider*, or None. Ollama legitimately has none."""
        if provider == "anthropic":
            return self.anthropic_api_key or _env("ANTHROPIC_API_KEY")
        if self.llm_api_key and provider == self.llm_provider:
            return self.llm_api_key
        preset = OPENAI_COMPAT_PRESETS.get(provider)
        if not preset:
            return None
        env_name = preset.get("env") or ""
        return _env(env_name) if env_name else None

    def endpoint_for(self, provider: str) -> tuple[str, str]:
        """(base_url, model) for an OpenAI-compatible provider.

        An explicit LEA_LLM_BASE_URL / LEA_MODEL always wins, so an unknown or
        self-hosted endpoint needs no code change.
        """
        preset = OPENAI_COMPAT_PRESETS.get(provider, {})
        base = self.llm_base_url or preset.get("base_url", "")
        model = self.model if self.model != DEFAULT_MODEL else preset.get("model", self.model)
        return base, model

    def is_openai_compat(self, provider: Optional[str] = None) -> bool:
        p = provider or self.resolved_llm()
        return p in OPENAI_COMPAT_PRESETS or bool(self.llm_base_url and p not in ("anthropic", "stub"))

    def resolved_search(self) -> str:
        if self.search_provider != "auto":
            return self.search_provider
        if self.offline:
            return "fixture"
        if self.tavily_api_key:
            return "tavily"
        if self.serper_api_key:
            return "serper"
        if self.brave_api_key:
            return "brave"
        return "duckduckgo"
