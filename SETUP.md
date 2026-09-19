# What you need to fill in

Everything below is a placeholder shipped with the repo. The project **runs
without any of it** — `lea enrich "Vanta" --offline` works on a cold clone with
no keys and no network. Fill these in to turn it from a working demo into your
demo.

Ordered by how much each one changes the result.

---

## 1. A model key — free options exist, and one of them is the default

The agent runs without any model key: the planner falls back to `StubClient`, a
deterministic heuristic. It genuinely searches, fetches, extracts and scores —
but its research policy is close to fixed, and on companies it has no recorded
fixtures for it tends to return `unknown` for industry, HQ and headcount.

A real model fixes both. You have three routes.

### Free — Google AI Studio (the documented default)

No credit card, no spend. Get a key at <https://aistudio.google.com/apikey>.

```bash
cp .env.example .env
# set GOOGLE_API_KEY=...
./.venv/bin/lea doctor        # confirms which provider resolved
```

**Free-tier volume is the real constraint, not cost.** Google allows roughly
**20 requests per day per model**. One company costs ~11 calls, so that is about
**two companies per day per model**. The quota is per *model*, and several Flash
variants exist, so switching `LEA_MODEL` when one is exhausted gives you more
headroom — but this tier suits demos, not batch runs. For a real list use Groq
(much higher free limits) or a small Anthropic balance. A per-day quota is not
retried; the error names the fix.

Also supported, same adapter, all free tiers: `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`CEREBRAS_API_KEY`, or a local Ollama at `http://localhost:11434/v1` needing no key
at all. Pick with `--provider`, or let `auto` take the first key it finds.

Caveat worth knowing: free models vary in how well they honour a forced JSON
schema. The adapter falls back through function-calling → JSON mode → a repair
loop, so output stays valid, but the ICP reasoning and outreach hooks read
flatter than a frontier model produces.

Free model lineups rotate. If a default model name 404s, run
`./.venv/bin/lea doctor --models` to list what your key actually serves.

### Paid — Anthropic, if you want the best output

Billed pay-as-you-go at <https://console.anthropic.com/>, **separately from any
Claude Pro/Max subscription** — a subscription includes no API credits.

Measured on a real run: one company is ~22,400 input and ~4,000 output tokens
(7 planning decisions + 3 synthesis calls).

| Model | Per company | 100 companies | 1,000 |
|---|---|---|---|
| Haiku 4.5 | $0.04 | $4.22 | $42 |
| Sonnet 5 *(default)* | $0.08 | $8.44 | $84 |
| Opus 5 | $0.21 | $21.10 | $211 |

Demoing a portfolio project costs under $2. Set `ANTHROPIC_API_KEY`; it wins over
the free providers when both are present.

### Free and offline — change nothing

`lea enrich "Vanta" --offline` works on a cold clone with no key and no network.
Good enough to show the loop; not good enough to claim the agent reasons.

## 2. A search key — optional, fixes the demo-day failure mode

**Where:** `.env`, one of `TAVILY_API_KEY` / `SERPER_API_KEY` / `BRAVE_API_KEY`.
**Why it matters:** the keyless DuckDuckGo default works, but DDG bot-blocks a
client after a handful of queries in quick succession — precisely what happens
when you run the demo three times on a call. Tavily's free tier removes the risk.
Auto-detected in the order Tavily → Serper → Brave; no code change needed.

## 3. Your real ICP

**Where:** `icp.yaml` in the project root (gitignored; start from `icp.example.yaml`).
**Currently:** a clearly-labelled `PLACEHOLDER PROFILE` modelling a generic B2B
SaaS vendor selling to GTM teams. Every score you see until you replace it is
scored against a profile that is not yours.

Two ways to replace it:

```bash
# derive one from accounts you actually won (needs ANTHROPIC_API_KEY)
./.venv/bin/lea icp infer --examples "Customer A, Customer B, Customer C" --out icp.yaml

# or start from the placeholder and edit the weights by hand
cp icp.example.yaml icp.yaml
./.venv/bin/lea icp show
```

## 4. Your target list

**Where:** `examples/companies.csv` — replace with your own.
(`examples/offline-demo.csv` is the two-company list behind the no-network demo;
leave it alone unless you record fixtures for your own companies.)
Any CSV with a `company`, `domain`, `name` or `website` column works, as does a
bare one-per-line list.

```bash
./.venv/bin/lea batch your-targets.csv --out out/
```

## 5. `LEA_CONTACT` — politeness

**Where:** `.env`. An email or URL you actually read. It goes in the User-Agent
of every outbound request so a site owner can reach you instead of guessing.
Blank means the crawler is anonymous, which is legal and rude.

## 6. `PROJECT_URL` — cosmetic

**Where:** `src/lea/config.py`, currently
`https://github.com/your-handle/lead-enrichment-agent`. Also appears in the
User-Agent. Point it at your repo once you publish.

---

## Known placeholder in the demo data

The **search-result fixtures** under `examples/fixtures/` are hand-authored, not
live captures — DuckDuckGo blocked the build machine's IP throughout
development. They use real DDG markup and are grounded in content from the real
pages, and each is tagged `"synthetic": true` so they are impossible to mistake
for recordings.

The **page fixtures** (Ramp and Vanta homepages, careers, about) *are* genuine
live recordings, and they are what the firmographics, signals and tech stack are
actually extracted from.

To replace the synthetic ones from an unblocked network:

```bash
LEA_RECORD_FIXTURES=1 ./.venv/bin/lea enrich "Vanta"
LEA_RECORD_FIXTURES=1 ./.venv/bin/lea enrich "Ramp"
```

Worth doing before you publish the repo, so nothing in it is synthetic.
