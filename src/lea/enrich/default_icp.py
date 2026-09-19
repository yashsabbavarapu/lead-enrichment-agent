"""The shipped fallback ICP.

This exists so `lea enrich "Ramp"` produces a real score on a cold install with
no config. It is deliberately a *placeholder* — the interesting path is
`lea icp infer --examples "..."`, which derives a rubric from your actual closed
won customers. The summary says so out loud, so nobody mistakes a demo default
for a researched profile.
"""

from __future__ import annotations

from lea.models import ICPCriterion, ICPProfile

DEFAULT_ICP_NAME = "default-b2b-saas-gtm"

DEFAULT_ICP_SUMMARY = (
    "PLACEHOLDER PROFILE — not derived from your customers. Models a B2B SaaS vendor "
    "selling workflow software to go-to-market teams (sales, marketing, RevOps) at "
    "venture-backed North American and Western European companies past product-market "
    "fit. Sweet spot: 50-500 employees, cloud-native stack, an in-house GTM org large "
    "enough to feel process pain, and a recent growth event (funding or GTM hiring) that "
    "creates budget and urgency. Run `lea icp infer --examples \"Customer A, Customer B, "
    "Customer C\"` to replace this with a rubric derived from real closed-won accounts."
)

DEFAULT_ICP = ICPProfile(
    name=DEFAULT_ICP_NAME,
    summary=DEFAULT_ICP_SUMMARY,
    criteria=[
        ICPCriterion(
            key="company_type",
            description="A B2B software or tech-enabled services company that sells to other businesses.",
            weight=2.0,
            rubric=(
                "10 = pure-play B2B SaaS with a self-described product and pricing page. "
                "7 = tech-enabled B2B services or B2B marketplace. "
                "4 = hybrid B2B/B2C. "
                "0 = consumer-only, public sector, non-profit, or a pure agency."
            ),
        ),
        ICPCriterion(
            key="gtm_segment",
            description="Sells into a market that requires a real outbound/inbound GTM motion, with an identifiable sales org.",
            weight=2.0,
            rubric=(
                "Grade from observable artefacts, not vibes: open AE/SDR/RevOps reqs, a "
                "CRM in the detected stack, a 'talk to sales' or demo-request path, sales "
                "tooling (Qualified, Outreach, Gong, 6sense). "
                "10 = open GTM reqs AND a CRM detected. "
                "7 = one of the two, or a clear enterprise sales motion. "
                "5 = evidence is absent either way - say so rather than guessing. "
                "3 = pure self-serve PLG with no sales headcount visible. "
                "0 = no commercial GTM function at all."
            ),
        ),
        ICPCriterion(
            key="company_size",
            description="Headcount in the 50-500 band, where GTM process pain is acute but procurement is still light.",
            weight=1.5,
            rubric=(
                "Grade on the best size evidence available, in this order. "
                "(a) A stated headcount or band: 10 = 51-200 or 201-500; 6 = 11-50 or "
                "501-1000; 3 = 1001-5000; 1 = 1-10 or 5000+. "
                "(b) No headcount but a public job board: grade on open-role count as a "
                "stage proxy - 9-10 = 20-59 live roles; 8-9 = 6-19; 6 = 1-5 or 60-149; "
                "3 = 150+. Say it is a hiring-volume inference and that exact headcount "
                "is unconfirmed. Do NOT state an employee range you did not read. "
                "(c) Neither: score 5 and say size is genuinely unknown. Do NOT fall back on "
                "what you already know about this company - if the gathered evidence does "
                "not contain a headcount or a job board, it is unknown, and guessing from "
                "memory is the failure this rubric exists to prevent."
            ),
        ),
        ICPCriterion(
            key="tech_maturity",
            description="Cloud-native, modern data/GTM stack, and evidence they buy rather than build tooling.",
            weight=1.5,
            rubric=(
                "10 = modern CRM (Salesforce/HubSpot), a data warehouse, and 2+ best-of-breed SaaS tools detected. "
                "6 = a modern CRM or analytics stack visible. "
                "3 = legacy or minimal stack. "
                "0 = strong build-it-ourselves signal (in-house everything, no third-party tooling)."
            ),
        ),
        ICPCriterion(
            key="growth_signals",
            description="A recent trigger event that creates budget and urgency: funding, GTM hiring, expansion, or a product launch.",
            weight=1.5,
            rubric=(
                "Weight every signal by how old it is - recency is most of the value here. "
                "10 = a signal under 30 days old AND open GTM roles right now. "
                "8 = a signal under 90 days old. "
                "6 = a dated signal 3-12 months old, or strong undated evidence. "
                "4 = the only signals are 1-2 years old. "
                "2 = everything is over 2 years old, or nothing was found. "
                "0 = layoffs, a down round, or an announced acquisition. "
                "An undated signal counts, but less than a dated recent one - a missing "
                "date is not evidence of staleness. Name the signals you used and their "
                "ages in the reasoning."
            ),
        ),
        ICPCriterion(
            key="geo",
            description="Headquartered in a market we can support and sell into today.",
            weight=1.0,
            rubric=(
                "10 = US, Canada, UK, or Western Europe. "
                "6 = ANZ, Nordics, or Israel. "
                "3 = elsewhere with an English-language site. "
                "0 = a market with a data-residency or language barrier we cannot serve."
            ),
        ),
    ],
    disqualifiers=[
        "Consumer-only (B2C) company with no business-facing product or sales motion.",
        "Direct competitor: sells GTM/sales-engagement/lead-enrichment software itself.",
        "Publicly announced acquisition, wind-down, or major layoffs in the last 6 months.",
    ],
    derived_from=[],
)
