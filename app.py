"""
Research Agent — Streamlit app using Groq (Llama 3.3 70B) plus web search.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import streamlit as st
from dotenv import load_dotenv
from ddgs import DDGS
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


def get_groq_api_key() -> str:
    """Local: `.env` / `GROQ_API_KEY` env var. Streamlit Community Cloud: app Secrets (same key name)."""
    k = (os.getenv("GROQ_API_KEY") or "").strip()
    if k:
        return k
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


def get_client() -> Groq:
    key = get_groq_api_key()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Use a `.env` file locally, or Streamlit Cloud → app → ⋮ → Settings → Secrets."
        )
    return Groq(api_key=key)


def groq_chat(client: Groq, system: str, user: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    msg = completion.choices[0].message
    return (msg.content or "").strip()


def step_analyze(client: Groq, topic: str) -> str:
    system = (
        "You are a senior market research lead. Your job is to plan research, not write the final report. "
        "Be specific and actionable."
    )
    user = f"""Topic (company or product): {topic}

Produce a concise research plan with:
- 3–6 key research questions we must answer
- Which angles matter most (market, product, pricing, competitors, risks, recent events)
- Suggested search themes (short phrases) we should use to gather evidence

Use clear bullet points. Do not invent facts about the company; only plan what to investigate."""
    return groq_chat(client, system, user)


def _search_queries(topic: str) -> OrderedDict[str, str]:
    """Build focused queries; extra disambiguation for short single-token names (e.g. Island vs islands)."""
    t = topic.strip()
    pairs: list[tuple[str, str]] = []
    # One-word (or compact) names match geography, common nouns, etc. — bias toward software/business context first.
    single_token = " " not in t and 2 <= len(t) <= 32
    if single_token:
        pairs.extend(
            [
                (f'"{t}" software company', "company_disambig"),
                (f"{t} B2B enterprise software company", "company_context"),
            ]
        )
    pairs.extend(
        [
            (f"{t} company", "company"),
            (f"{t} product", "product"),
            (f"{t} pricing", "pricing"),
            (f"{t} competitors", "competition"),
            (f"{t} company news", "news"),
        ]
    )
    out: OrderedDict[str, str] = OrderedDict()
    for q, label in pairs:
        if q not in out:
            out[q] = label
    return out


def gather_search_context(topic: str, max_per_query: int = 8) -> str:
    """Collect snippets from the open web (DuckDuckGo)."""
    queries = _search_queries(topic)
    blocks: list[str] = []
    try:
        ddgs = DDGS()
        for q, label in queries.items():
            hits = list(ddgs.text(q, max_results=max_per_query))
            lines = [f"[{label}] {q}"]
            for h in hits:
                title = h.get("title", "")
                body = h.get("body", "")
                url = h.get("href", "")
                lines.append(f"- {title}: {body} ({url})")
            blocks.append("\n".join(lines))
    except Exception as e:
        blocks.append(f"Search error (some results may be missing): {e}")
    return "\n\n".join(blocks)


def step_synthesize(client: Groq, topic: str, plan: str, search_blob: str) -> str:
    system = (
        "You are a research analyst writing an executive briefing. "
        "Ground every claim in the provided search excerpts when possible. "
        "If evidence is thin or conflicting, say so explicitly. "
        "Do not fabricate prices, dates, or news events."
    )
    user = f"""Topic: {topic}

--- Research plan (from prior analysis) ---
{plan}

--- Web search excerpts (may be incomplete or noisy) ---
{search_blob}

Write a structured report with EXACTLY these Markdown section headers (in this order):

## Overview
## Pricing
## Strengths
## Weaknesses
## Recent News

Under each section, use bullets or short paragraphs. Cite sources inline like [source: domain or title] when the claim comes from search excerpts. If a section has no reliable evidence, write "Insufficient evidence in retrieved sources." and suggest what to verify next."""
    return groq_chat(client, system, user)


def main() -> None:
    st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
    st.title("Research Agent")
    st.caption("Multi-step research: plan → web search → synthesis (Groq **llama-3.3-70b-versatile**).")

    topic = st.text_input(
        "Company or product name",
        placeholder="e.g. Acme Corp, Notion, AWS Lambda",
    )

    api_key = get_groq_api_key()
    if not api_key:
        st.error(
            "Missing `GROQ_API_KEY`. **Locally:** add it to `.env` in the project root and restart. "
            "**On Streamlit Cloud:** App → ⋮ → Settings → Secrets, add `GROQ_API_KEY = \"...\"` in TOML format, then reboot the app."
        )

    run = st.button("Run research", type="primary", disabled=not topic.strip() or not api_key)

    t = topic.strip()
    cached = (
        t
        and st.session_state.get("last_topic") == t
        and st.session_state.get("last_report")
    )

    if run:
        client = get_client()
        progress = st.status("Running research pipeline…", expanded=True)
        try:
            with progress:
                progress.write("Step 1: Analyzing what to research…")
                plan = step_analyze(client, t)
                st.session_state["last_plan"] = plan

                progress.write("Step 2: Searching the web (competitive positioning, pricing, news)…")
                search_blob = gather_search_context(t)
                st.session_state["last_search_context"] = search_blob

                progress.write("Step 3: Synthesizing structured report…")
                report = step_synthesize(client, t, plan, search_blob)
                st.session_state["last_report"] = report
                st.session_state["last_topic"] = t

                progress.update(label="Done", state="complete")
        except Exception as e:
            progress.update(label="Failed", state="error")
            st.error(f"Research run failed: {e}")
            return
    elif not cached:
        return

    if cached and not run:
        st.info("Showing the last run for this topic. Change the name or click **Run research** to refresh.")

    plan = st.session_state["last_plan"]
    search_blob = st.session_state["last_search_context"]
    report = st.session_state["last_report"]

    st.subheader("Research plan")
    st.markdown(plan)

    with st.expander("Raw search context (for transparency)"):
        st.text_area("Search excerpts", search_blob, height=320, label_visibility="collapsed")

    st.subheader("Report")
    st.markdown(report)

    st.download_button(
        "Download report (.md)",
        data=f"# Research: {t}\n\n{report}",
        file_name=f"research-{t.replace('/', '-')[:80]}.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()
