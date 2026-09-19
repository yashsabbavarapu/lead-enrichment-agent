# lead-enrichment-agent

An autonomous research agent that enriches B2B leads. Give it a company name and it **decides its own research path** — search, read the one page that looks most promising, notice what is still missing, then search again with a different query shaped by what it just learned — and returns a structured record, an ICP fit score with per-criterion reasoning, grounded outreach hooks, and the full trace of how it got there. A conventional enrichment pipeline runs the same fixed sequence of API calls for every company and returns nulls where the data was not in the vendor's database. This one plans, adapts when a step fails, cites a URL for every claim, and tells you what it could not find out.

Runs end-to-end with **zero API keys**.

## What makes this an agent, not a script

- **Dynamic tool selection.** Each turn the planner picks from `web_search`, `fetch_page`, `detect_tech`, `finish` based on the current gap list. Two companies rarely produce the same sequence of calls.
- **Budget-aware planning.** The planner sees its remaining step, search and fetch budget in-context, so it spends the last two steps closing the highest-value gap instead of wandering.
- **A redundancy guard.** Repeating a query or re-fetching a URL is intercepted, and the *failure is fed back into the next decision* as an observation ("that query returned the same three results — try different terms"). The agent changes tack rather than looping.
- **Gap-driven stopping.** It stops when the priority fields are filled or genuinely unobtainable, not when a counter hits a limit. `stop_reason` records which it was.
- **A replayable trace.** Every plan → act → observe cycle is captured with the thought, the tool call, the observation and its latency. `lea replay out/ramp.json` re-renders a past run step by step.

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"

./.venv/bin/lea doctor                                  # which providers resolved, and why

# --offline replays recorded fixtures: no network, no keys, always works.
./.venv/bin/lea enrich "Ramp" --offline                 # one company, live trace in the terminal
./.venv/bin/lea batch examples/offline-demo.csv --offline # a list -> a spreadsheet-ready CSV
./.venv/bin/lea replay out/ramp.json                    # re-render a saved run

# Drop --offline to research live off the public web (see the caveat below).
./.venv/bin/lea enrich "Vanta"
./.venv/bin/lea icp infer --examples "Ramp, Vanta, Retool"   # derive a rubric from real customers
```

> Two target lists ship with the repo. `examples/offline-demo.csv` holds the two companies
> with recorded page fixtures, so it runs fully offline. `examples/companies.csv` is the
> realistic 8-company list with a mix of strong fits and deliberate misses — it needs live
> search or a search key, and offline it will correctly blank 6 of 8.

**No keys required — with one honest caveat about live search.**

- `--offline` is the **guaranteed** path. It serves every HTTP call from `examples/fixtures/` and always works: no network, no rate limits, no flakiness. Use it to evaluate this project, and in CI. Every example in this README comes from an `--offline` run.
- **Live keyless search works, but can be rate-limited.** The default provider is DuckDuckGo with no key. It bot-blocks after a handful of queries, returning 429 or an empty result set — it was blocking our own build machine outright while this was written. The agent degrades gracefully (the failed search becomes an observation and it re-plans), but a batch run without a search key will be unreliable.
- **A search key removes that risk entirely.** `TAVILY_API_KEY`, `SERPER_API_KEY` or `BRAVE_API_KEY` — any one of them — and live search becomes dependable.
- **A model key is optional, and free options exist.** Without one the planner is a deterministic stub following a sensible research policy, and ICP scoring falls back to a deterministic rubric. With one, planning and extraction become real reasoning: the agent chooses genuinely non-obvious next steps and the record gets noticeably richer. The planner talks to an `LLMClient` with two methods, so the provider is a swap, not a rewrite:

  | Provider | Cost | Set |
  |---|---|---|
  | Google AI Studio (Gemini) | free | `GOOGLE_API_KEY` |
  | Groq | free | `GROQ_API_KEY` |
  | OpenRouter | free tier | `OPENROUTER_API_KEY` |
  | Cerebras | free tier | `CEREBRAS_API_KEY` |
  | Ollama (local) | free | nothing — `--provider ollama` |
  | Anthropic | paid, best output | `ANTHROPIC_API_KEY` |

  Any other OpenAI-compatible endpoint works via `LEA_LLM_BASE_URL`, no code change. `lea doctor` shows which resolved and why.

> **Free tiers limit volume, not just cost.** Google AI Studio allows ~20 requests per day *per model*, and one company costs ~11 calls — about two companies per day per model. Fine for demos; use Groq or a small paid balance for batch runs. The quota is per model, so switching `LEA_MODEL` buys more headroom.

> **Model ids go stale.** Free lineups rotate, and each provider's default model id is an educated guess that may already be retired. If calls fail, run `lea doctor --models` to list what your key actually serves, then set `LEA_MODEL`. The OpenAI-compatible adapter is tested against mocked endpoints but has **not** been exercised against a live one — the first real key is the first real proof.

Keys are picked up automatically; `lea doctor` prints exactly which provider resolved and why. See **[SETUP.md](SETUP.md)** for the placeholder checklist and key setup.

A note on the fixtures, since it affects how you read the demo: the **page fixtures are real recordings** of live pages, while the **search-result fixtures are synthetic** — hand-written to give the agent a stable, sensible set of links to choose between. So an offline run exercises real extraction over real HTML, but the search step is a fixed script rather than a live index.

## ICP scoring

The interesting part is that you do not write the rubric. Name three to five accounts you have already won:

```bash
lea icp infer --examples "Ramp, Vanta, Retool" --out icp.yaml
```

The model articulates what those customers have in common as a set of weighted, observable criteria plus two or three hard disqualifiers. The result is plain YAML with a header comment explaining every field — edit the weights by hand, commit it, re-run. Without an `icp.yaml`, a clearly-labelled placeholder profile ships in the box so a fresh install still produces a real score.

Scoring produces a 0–10 per criterion with written reasoning and cited evidence URLs. Tiers: **A** ≥ 80, **B** ≥ 60, **C** ≥ 40, else **D**. Any disqualifier hit forces tier D and caps the score regardless of how well the company scores elsewhere. `confidence` drops when criteria were graded without evidence — a score built on three unknowns says so on its face.

### A company it could not research reports no assessment, not a middling score

If research establishes nothing — no domain, industry, headcount, signals or tech — the output is score 0, tier D, confidence 0, and a summary stating in words that no fit could be assessed. It is deliberately *not* the neutral 50/C a half-finished scorer would emit, because 50/C reads as a considered judgement about the company and someone will work the list believing it was one. The model is not called at all in that case; there is nothing for it to reason about. Genuinely partial records still score normally — that is exactly what `confidence` is for.

The same instinct governs the two fields that head the report:

- A page `<title>` is rejected rather than rendered as the company name. `"About Us - Different By Design"` does not become the name; the record falls back to a name derived from the target or the domain's registrable label. Brand furniture is stripped, so `"Ramp | Corporate Cards"` does become `Ramp`.
- An aggregator host is never accepted as the company's own domain — Crunchbase, LinkedIn, Tracxn, PitchBook, ZoomInfo and the rest, editable as `AGGREGATOR_DOMAINS`. The denylist is overridden when the host really is the company, so GitHub still resolves to `github.com`. A host resembling neither the target nor the company name is left unknown rather than misattributed.
- Only URLs that actually returned usable text are cited in `sources`. A 404 or robots-blocked fetch is evidence of nothing and is not a source.

## Example trace

Real output from `./.venv/bin/lea enrich "Ramp" --offline`, trimmed for length — observation bodies shortened, `#` lines added as annotation. Run the command and you should get this shape.

```
researching Ramp   [llm=stub search=fixture offline]

  1 > search  ok  21ms
      Nothing is known yet about Ramp. A broad overview search should surface
      the official site and a one-line description to anchor everything else.
      web_search("Ramp company overview")
      | 1. Spend Management, Corporate Cards & Accounts Payable Solutions |
      | Ramp - https://ramp.com/ ...
      | https://ramp.com/, https://ramp.com/about-us (+2 more)

  2 v fetch   ok  8ms
      The search results point at https://ramp.com/ as Ramp's own site.
      Firmographics (what they sell, size, HQ, founding year) belong on the
      company's own pages.
      fetch_page("https://ramp.com/")
      | Spend Management, Corporate Cards & Accounts Payable Solutions | Ramp ...

  3 # tech    ok  11ms
      Ramp's homepage is fetched but its stack is unknown. Detected technology
      is a cheap, concrete qualification signal and a hook to lean on later.
      detect_tech("https://ramp.com/")
      | - [framework] Next.js (0.95) - header 'x-powered-by: Next.js'
      | - [hosting]   Vercel  (0.95) - header 'x-vercel-id: ...'

  4 > search  ok  21ms
      #  <- the agent changes objective here. Firmographics are covered, so it
      #     stops reading the site and goes hunting for a dated event instead.
      Firmographics are covered but there is no dated signal yet. Funding,
      hiring and news are what an outreach hook can actually lean on.
      web_search("Ramp funding OR hiring OR news 2025")
      | 1. Careers - Change How Businesses Run | Ramp - https://ramp.com/careers ...

  5 v fetch   ok  8ms
      https://ramp.com/about-us looks like the highest-value unread page (news /
      about / careers) for turning the search snippets into a citable, dated signal.
      fetch_page("https://ramp.com/about-us")

  6 v fetch   ok  7ms
      https://ramp.com/careers looks like the highest-value unread page ...
      fetch_page("https://ramp.com/careers")

  7 * synth   ok
      #  <- gap-driven stop: 2 of 12 steps and 3 of 5 searches still unspent.
      Firmographics, technology and signal searches are all done and no unread
      page looks worth the remaining budget. Stopping here rather than padding
      the trace.
      finish()

  8 * synth   ok  10ms   | resolved 4 core field(s); 5 signal(s), 4 tech entries.
  9 * synth   ok   5ms   | tier B - 67/100 across 6 criteria.
 10 * synth   ok   6ms   | 2 hook(s) kept.

10 steps - 2 searches - 4 pages - 0.2s - llm=stub search=fixture
outcome: COMPLETE - complete: agent called finish
wrote out/ramp.json
wrote out/ramp.trace.md
```

Steps 4 and 7 are the ones worth reading. A fixed pipeline fetches a preset list and stops when the list runs out. This run switched from firmographics to signal-hunting because the first objective was met, then stopped with budget still on the table because nothing unread justified spending it.

The per-criterion scoring from that same run — this reasoning is generated from the record, not canned:

```
  criterion        score    wt   reasoning
 ----------------------------------------------------------------------------
  company_type       8.0     2   Industry/description mentions 'fintech',
                                 consistent with a B2B product.
  gtm_segment        5.0     2   The company is actively hiring - "See open
                                 positions".
  company_size       5.0   1.5   Headcount could not be established from public
                                 sources, so this is scored mid rather than
                                 penalised. It is a gap, not a negative.
  tech_maturity      6.6   1.5   Detected 3 technologies across 3 categories;
                                 cloud-native infrastructure (Cloudflare and
                                 Vercel).
  growth_signals     7.0   1.5   Trigger events found - hiring: "See open
                                 positions"; 3 product launches; 1
                                 leadership/partnership signal.
  geo               10.0     1   Headquartered in New York, United States - a
                                 core market we sell into today.

  -> tier B - 67.3/100 - confidence 0.85 - unknowns: employee_range
```

Note `company_size` scoring 5.0 with the reason stated rather than 0. An unknown is not a negative, and `confidence` is where missing evidence is accounted for.

With no model key these scores come from the deterministic rubric described above, which grades each criterion off the record. It is not model reasoning and the summary line says so — but two different companies get two different explainable scores instead of a constant.

## Example output

The record from that same `--offline` run (`out/ramp.json`, trimmed):

```json
{
  "input": "Ramp",
  "company_name": "Ramp",
  "domain": "ramp.com",
  "description": "Make expense management easy with Ramp's spend management platform.",
  "industry": "Fintech",
  "employee_range": null,
  "hq_city": "New York",
  "hq_country": "United States",
  "founded_year": 2019,
  "signals": [
    {
      "kind": "hiring",
      "headline": "See open positions",
      "detail": "Reported June 2025.",
      "date": "June 2025",
      "source_url": "https://ramp.com/about-us",
      "recency_note": "over a year ago"
    },
    {
      "kind": "product",
      "headline": "Introducing Stack by Ramp.",
      "date": null,
      "source_url": "https://ramp.com/",
      "recency_note": "undated"
    }
  ],
  "tech_stack": [
    { "category": "framework", "name": "Next.js", "evidence": "header 'x-powered-by: Next.js'", "confidence": 0.95 },
    { "category": "hosting",   "name": "Vercel",  "evidence": "header 'x-vercel-id: iad1::...'", "confidence": 0.95 },
    { "category": "infra",     "name": "Cloudflare", "evidence": "header 'server: cloudflare'", "confidence": 0.95 }
  ],
  "icp_fit": {
    "score": 67.3,
    "tier": "B",
    "confidence": 0.85,
    "summary": "Tier B (67/100). Strongest: company_type 8/10, growth_signals 7/10. Graded by the deterministic rubric (no LLM judgement available) - set ANTHROPIC_API_KEY for reasoned per-criterion scoring."
  },
  "hooks": [
    {
      "angle": "stack choice",
      "hook": "Next.js, Vercel and Cloudflare is a deliberate stack, not an accident - someone there has opinions about GTM data. Those are usually the teams that notice the seam between them first.",
      "grounded_in": "framework Next.js",
      "source_url": "https://ramp.com/"
    }
  ],
  "unknowns": ["employee_range"],
  "sources": [
    "https://ramp.com/",
    "https://ramp.com/about-us",
    "https://ramp.com/careers",
    "https://www.crunchbase.com/organization/ramp-6b26",
    "https://techcrunch.com/2025/03/ramp-funding-round/"
  ]
}
```

Two things in there are deliberate rather than broken. `"employee_range": null` with `"unknowns": ["employee_range"]` is the agent declining to guess a headcount it could not source. And the hooks lean on a stack choice and a hiring line because that is what this evidence actually supports — there is no invented funding round, because the fixtures contain no funding announcement the extractor could date.

## Architecture

```
src/lea/
  models.py          Pydantic contracts. Imports nothing from lea.* — the leaf of the graph.
  config.py          Settings; resolves providers from env. Keys upgrade, never gate.
  tools/             web_search, fetch_page, detect_tech. Return Observations, never raise.
  llm/               LLMClient protocol + anthropic and stub providers. decide() and extract().
  prompts.py         Every system prompt, in one file, versioned with the code.
  agent/             The plan -> act -> observe loop: budget, redundancy guard, stop conditions.
  trace.py           Trace capture, rendering and JSON replay.
  enrich/
    extract.py       Evidence -> normalized CompanyRecord. Dedupe, bucket, range-check, merge.
    icp.py           infer_icp / score_icp / YAML load+save. The arithmetic lives here.
    default_icp.py   The shipped placeholder profile.
    hooks.py         Hook generation + the mechanical anti-hallucination filter.
  io_csv.py          Forgiving CSV in, flat spreadsheet-friendly CSV out.
  cli.py             enrich / batch / icp / replay / doctor.
```

Design decisions worth naming:

- **The model is a swap, not a dependency.** `LLMClient` is two methods, so Anthropic, any OpenAI-compatible endpoint and a keyless deterministic planner are peers behind one interface. The OpenAI-compatible adapter degrades through forced function-calling → JSON mode → a prose repair loop, because free models vary wildly in how well they honour a schema and the agent should produce weaker output rather than none.
- **The LLM never does arithmetic on the score.** It grades each criterion 0–10 and writes the reasoning; the weighted roll-up, the tier thresholds and the disqualifier override are computed in Python. A model that quietly adds wrong is far more dangerous than one that reasons badly, because nothing about the output looks off — you only find out after someone has worked the list.
- **The model's weights are discarded at scoring time.** Criterion weights always come from the profile, so the model cannot silently reweight the rubric it was asked to apply.
- **Tools never raise into the loop.** Every tool returns an `Observation` with `ok=False` and an error string. A 404 is not an exception, it is evidence — and it has to reach the planner as context so the next decision can react to it. Exceptions would abort the run; observations change its mind.
- **Evidence is cited, and citations are filtered.** Every signal, tech entry and criterion score carries the URL it came from, and any URL the model cites that was not actually fetched is stripped before it reaches the record. A claim you cannot trace is a claim a rep should not repeat.
- **Extraction output is post-processed, not trusted.** Domains are normalized and screened against an aggregator denylist, company names are rejected when they look like page titles, headcounts are bucketed, founding years outside 1800–today are dropped, signals are deduped by normalized headline and sorted most-recent-first, tech is deduped keeping the highest confidence. These normalizers are the last line of defence before a record is rendered and saved, so they assume the model's output is wrong until it passes.
- **Refusing to answer is a supported outcome.** An unresearched company gets an explicit non-assessment, an unverifiable domain stays `null`, and a hook with no real fact behind it is dropped. A blank field routes the company to a human; a plausible wrong one does not.
- **Hooks are mechanically filtered.** Anything over 45 words, containing a cliché from a blocklist, or whose `grounded_in` does not match a fact actually present in the record is dropped. If that leaves fewer than two, deterministic hooks are built from the strongest real signal — plain, but true.
- **The fetcher assumes its own URLs are hostile.** The agent follows URLs an LLM chose out of untrusted search results, which makes it a confused deputy by construction. Every URL and every redirect hop is resolved and refused if it lands on loopback, RFC1918, link-local or a cloud metadata endpoint — so a page that redirects to `169.254.169.254` gets an error, not credentials. The residual DNS-rebinding window is commented in `tools/http.py` rather than pretended away.
- **Unknowns are a first-class field.** Honest gaps are more useful than confident guesses, and they are what tells you which companies deserve a human pass.

## Limitations

- **Keyless live search is rate-limited, not merely worse.** DuckDuckGo without a key bot-blocks after a handful of queries. `--offline` always works and a search key always works; live keyless sits between them and should not be relied on for a batch run.
- **The offline search fixtures are synthetic.** The page fixtures are real recordings, but the search results are hand-written. An offline run shows real extraction over real HTML with a scripted set of links — it demonstrates the loop honestly, but it is not evidence that live search returns those same links.
- **The stub planner is a policy, not a planner.** With no model key at all, tool selection follows a fixed heuristic. It runs and it demonstrates the loop, but genuinely adaptive behaviour needs a real model — and a free one is enough to get it.
- **Keyless ICP scoring is a rubric, not judgement.** Without any model key, criteria are graded by the deterministic scorer in `enrich/icp.py` — keyword and count heuristics over the record. It discriminates between companies and explains itself, but it cannot read nuance the way a model does, and the summary line says which mode produced the score. Its segment inference is only as good as the extracted `industry` string.
- **Headcount is a bucket, not a number.** Public sources disagree constantly; `employee_estimate` is a bucket midpoint and should be treated as an order of magnitude.
- **Tech detection is shallow.** It reads public page markup and job postings. It sees front-end and analytics tooling well, back-end and data infrastructure barely at all, and it infers rather than asserts — hence the per-entry confidence.
- **ICP inference reflects the examples you give it.** Three examples from one vertical produce a rubric that overfits to that vertical. It is a starting draft to edit, not an oracle.
- **No CRM writeback and no dedupe against existing records.** This is a portfolio project, not a production data platform. Search does pace itself, rotate endpoints and back off, but there is no persistent cross-run rate-limit budget.
- **Public web only.** No LinkedIn, no paid data vendors, no anything behind a login. `robots.txt` is respected by default.
