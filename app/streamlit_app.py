from __future__ import annotations

import json
import html
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.graph.workflow import ask


st.set_page_config(
    page_title="Cosmetic Disclosure Search",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink: #172824; --muted: #64746e; --green: #176b55; --mint: #dcefe7; --orange: #c75b32; --line: #d8e2dc; --paper: #fbfcf8; }
html, body, [class*="css"] { font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(ellipse at 8% 2%, rgba(221,239,230,.76), transparent 32%), linear-gradient(150deg,#fbfcf8 0%,#eef4f0 58%,#f9f7ef 100%); }
[data-testid="stSidebar"] { background: #172824; }
[data-testid="stSidebar"] * { color: #eef5f0; }
.eyebrow { font: 500 11px Consolas, monospace; letter-spacing: 0; text-transform: uppercase; color: var(--green); }
.app-title { font: 700 44px/1.08 Georgia, 'Times New Roman', serif; margin: 9px 0 10px; color: var(--ink); }
.lead { color: var(--muted); font-size: 15px; max-width: 780px; margin-bottom: 24px; line-height: 1.55; }
.answer-panel { background: rgba(255,255,255,.82); border-left: 4px solid var(--green); padding: 22px 24px; border-radius: 4px; box-shadow: 0 8px 28px rgba(23,40,36,.06); }
.answer-panel p { margin: 0; font-size: 17px; line-height: 1.65; }
.question-highlight { background: #223631; color: #f3f7f2; border-left: 4px solid #77bda0; border-radius: 4px; padding: 16px 18px; margin: 8px 0 14px; font-size: 15px; line-height: 1.55; }
.question-highlight strong { display: block; color: #9bd2b8; font: 11px Consolas, monospace; margin-bottom: 6px; }
.turn-heading { font: 600 15px 'Trebuchet MS', sans-serif; color: var(--ink); }
.small-label { color: var(--muted); font: 11px Consolas, monospace; text-transform: uppercase; letter-spacing: 0; }
.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); color: white; }
div[data-testid="stForm"] { background: rgba(255,255,255,.72); padding: 16px 18px 6px; border: 1px solid var(--line); border-radius: 5px; }
[data-testid="stMetric"] { background: rgba(255,255,255,.58); border-top: 2px solid var(--green); padding: 12px 14px; }
code, pre { font-family: Consolas, monospace !important; }
hr { border-color: var(--line); }
@media (max-width: 640px) { .app-title { font-size: 32px; } .answer-panel { padding: 17px; } }
</style>
""", unsafe_allow_html=True)


def ollama_status() -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, [item["name"] for item in payload.get("models", [])]
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False, []


available, installed_models = ollama_status()

with st.sidebar:
    st.markdown("<div class='eyebrow' style='color:#91c7ae'>LOCAL DISCLOSURE POC</div>", unsafe_allow_html=True)
    st.markdown("### Runtime")
    if available:
        st.success("Ollama is reachable")
    else:
        st.warning("Ollama is offline. Structured and vector search still work; answers use deterministic synthesis.")
    model_present = any(name.split(":")[0] == OLLAMA_MODEL.split(":")[0] for name in installed_models)
    st.caption(f"Answer model: {OLLAMA_MODEL} · {'ready' if model_present else 'not listed'}")
    st.caption("Embeddings: nomic-embed-text · local")
    use_local_model = st.toggle("Use local model for summaries", value=available and model_present)
    result_limit = st.slider("Evidence rows", min_value=5, max_value=100, value=20, step=5)
    st.markdown("---")
    st.markdown("<span class='small-label'>DATA SOURCE</span>", unsafe_allow_html=True)
    st.caption("California Safe Cosmetics Program disclosures")
    st.caption("DuckDB facts · Qdrant vectors + BM25")
    if st.button("Clear conversation", width="stretch"):
        st.session_state["chat_history"] = []
        st.session_state.pop("pending_question", None)
        st.rerun()

st.session_state.setdefault("chat_history", [])

st.markdown("<div class='eyebrow'>CALIFORNIA SAFE COSMETICS PROGRAM · LOCAL RESEARCH CONSOLE</div>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Chemical disclosure search</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='lead'>Ask about reported ingredients, products, companies, categories, and lifecycle dates. "
    "Exact questions use SQL; uncertain chemical names combine local vectors and BM25 before the database verifies records.</div>",
    unsafe_allow_html=True,
)

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Question",
        placeholder="Which products contain CAS 75-07-0?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search disclosures", type="primary", use_container_width=False)

sample_questions = {
    "Question 1 · CAS lookup": "Which products contain CAS 75-07-0?",
    "Question 2 · Semantic chemical search": "Which products contain titanium oxide?",
    "Question 3 · Discontinued products": "Show products discontinued in 2020",
    "Question 4 · Brand comparison": 'Compare brands AVON and MARK for CAS 13463-67-7 in SubCategory "Lip Color - Lipsticks, Liners, and Pencils", discontinued in 2010.',
    "Question 5 · Reporting trend": "Summarize reporting trends for New Avon LLC",
}
with st.expander("Try a numbered test question", expanded=not st.session_state["chat_history"]):
    selected_question = st.selectbox("Choose a test question", list(sample_questions))
    selected_text = sample_questions[selected_question]
    st.markdown(
        f"<div class='question-highlight'><strong>{html.escape(selected_question)}</strong>{html.escape(selected_text)}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Load selected question", key="load_sample_question"):
        st.session_state["pending_question"] = selected_text
        st.rerun()

pending_question = st.session_state.pop("pending_question", None)
if pending_question:
    question = pending_question
    submitted = True

if submitted:
    if not question.strip():
        st.warning("Enter a question to search the dataset.")
    else:
        history = st.session_state["chat_history"]
        previous_result = history[-1]["result"] if history else None
        conversation_context = None
        if previous_result and previous_result.get("confidence") != "needs_clarification":
            conversation_context = {"query_plan": previous_result.get("query_plan", {})}
        with st.spinner("Checking local data and retrieval models…"):
            try:
                result = ask(
                    question,
                    limit=result_limit,
                    use_local_model=use_local_model,
                    conversation_context=conversation_context,
                )
                history.append({"question": question, "result": result})
                st.session_state["chat_history"] = history[-10:]
            except Exception as error:
                st.error(f"Search failed: {type(error).__name__}: {error}")

for turn_index, turn in enumerate(st.session_state["chat_history"]):
    result = turn["result"]
    plan = result.get("query_plan", {})
    usage = result.get("model_usage", {})
    title_text = turn["question"]
    if len(title_text) > 84:
        title_text = title_text[:81].rstrip() + "..."
    with st.expander(f"Question {turn_index + 1} · {title_text}", expanded=turn_index == len(st.session_state["chat_history"]) - 1):
        st.markdown(
            f"<div class='question-highlight'><strong>QUESTION {turn_index + 1}</strong>{html.escape(turn['question'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(result["answer"])
        summary = result.get("summary", {})
        counts = summary.get("counts", {})
        workflow_ms = sum(item.get("latency_ms", 0) for item in result.get("step_metrics", []))
        product_count = counts.get("product_count")
        ingredient_records = counts.get("ingredient_records")
        st.caption(
            f"{plan.get('intent', 'question').replace('_', ' ')} · "
            f"{plan.get('retrieval_mode', 'direct')} · "
            f"{product_count if product_count is not None else '—'} products · "
            f"{ingredient_records if ingredient_records is not None else '—'} records · "
            f"{workflow_ms:.0f} ms · {usage.get('total_tokens', 0)} local tokens · $0 API"
        )
        if result.get("warnings"):
            for warning in result["warnings"]:
                st.warning(warning)

        with st.expander("Performance · latency and token usage", expanded=False):
            metric_rows = []
            for step in result.get("step_metrics", []):
                is_synthesis = step.get("step") == "answer_synthesizer"
                metric_rows.append({
                    "Step": step.get("step"),
                    "Latency (ms)": step.get("latency_ms", 0),
                    "Prompt tokens": usage.get("prompt_tokens", 0) if is_synthesis else 0,
                    "Completion tokens": usage.get("completion_tokens", 0) if is_synthesis else 0,
                    "Total tokens": usage.get("total_tokens", 0) if is_synthesis else 0,
                    "API cost (USD)": usage.get("api_cost_usd", 0.0) if is_synthesis else 0.0,
                })
            if metric_rows:
                st.dataframe(
                    pd.DataFrame(metric_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
                        "API cost (USD)": st.column_config.NumberColumn(format="$%.4f"),
                    },
                )
            st.caption(
                f"Model: {usage.get('model', OLLAMA_MODEL)} · status: {usage.get('status', 'not used')} · "
                f"total workflow: {workflow_ms:.2f} ms · local API cost: $0.00"
            )

        with st.expander("Evidence", expanded=False):
            evidence = result.get("evidence", [])
            if evidence:
                st.dataframe(pd.DataFrame(evidence), width="stretch", hide_index=True)
            else:
                st.info("No row-level evidence was returned for this turn.")

        with st.expander("How this answer was derived", expanded=turn_index == len(st.session_state["chat_history"]) - 1):
            st.markdown("**Query plan**")
            st.json({
                "query_plan": plan,
                "confidence": result.get("confidence"),
                "model_usage": usage,
            })
            if result.get("step_metrics"):
                st.markdown("**Execution trace · exact work per step**")
                st.dataframe(pd.DataFrame(result["step_metrics"]), width="stretch", hide_index=True)
            aggregate = summary.get("aggregate", {})
            if aggregate.get("trend"):
                st.markdown("**Trend aggregate**")
                st.dataframe(pd.DataFrame(aggregate["trend"]), width="stretch", hide_index=True)
            if aggregate.get("chemicals"):
                st.markdown("**Chemical aggregate**")
                st.dataframe(pd.DataFrame(aggregate["chemicals"]), width="stretch", hide_index=True)