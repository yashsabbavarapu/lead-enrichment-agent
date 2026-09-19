# What you need to fill in

Everything below is a placeholder shipped with the repo. The project **runs
without any of it** — `lea enrich "Vanta" --offline` works on a cold clone with
no keys and no network. Fill these in to turn it from a working demo into your
demo.

Ordered by how much each one changes the result.

---

## 1. `ANTHROPIC_API_KEY` — the one that matters

**Where:** `.env` (copy from `.env.example`)
**Without it:** the planner falls back to `StubClient`, a deterministic heuristic
brain. It genuinely searches, fetches, detects tech, extracts and scores — the
whole pipeline runs — but its research policy is close to fixed, so the trace is
less impressive than the pitch implies.
**With it:** Claude chooses each next step from what it has just read. This is
what makes the "real agent, not a script" claim land in a demo.

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/lea doctor        # confirms it was picked up
```

## 2. A search key — optional, but fixes the demo-day failure mode

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
