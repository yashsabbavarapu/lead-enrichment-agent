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


@dataclass
class Settings:
    # --- providers -------------------------------------------------------
    llm_provider: str = "auto"  # auto | anthropic | stub
    model: str = DEFAULT_MODEL
    anthropic_api_key: Optional[str] = None
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
        if self.llm_provider != "auto":
            return self.llm_provider
        return "anthropic" if self.anthropic_api_key else "stub"

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
