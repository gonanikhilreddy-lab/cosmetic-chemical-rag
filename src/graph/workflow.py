from functools import lru_cache
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    answer_synthesizer_node,
    clarification_node,
    entity_extraction_node,
    evidence_builder_node,
    input_guardrail,
    out_of_scope_node,
    output_guardrail_node,
    planner_node,
    route_node,
    semantic_retrieval_node,
    structured_retrieval_node,
)
from src.schemas.agent_state import AgentState


def _step_details(name: str, state: AgentState, updates: dict[str, Any]) -> str:
    if name == "input_guardrail":
        return "Accepted dataset-related question." if not updates.get("out_of_scope") else "Rejected question outside dataset scope."
    if name == "planner":
        return f"Intent: {updates.get('intent')}; date field: {updates.get('date_field')}; range: {updates.get('date_from')} to {updates.get('date_to', 'open')} exclusive."
    if name == "entity_extraction":
        plan = updates.get("query_plan", {})
        return f"Entities: {updates.get('entities', {})}; inherited: {plan.get('inherited_entities', {})}; comparisons: {updates.get('comparisons', {})}; semantic hint: {updates.get('chemical_hint') or 'none'}."
    if name == "router":
        return f"Selected {updates.get('retrieval_mode')} route."
    if name == "semantic_retrieval":
        candidates = updates.get("semantic_candidates", [])[:3]
        rejected = updates.get("semantic_rejected", [])[:3]
        brief = [
            {
                "chemical": item.get("chemical_name"),
                "vector_score": item.get("vector_score"),
                "bm25_score": item.get("bm25_score"),
                "vector_rank": item.get("vector_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "relevance_score": item.get("relevance_score"),
            }
            for item in candidates
        ]
        return f"Fused vector K=10/BM25 K=10; reranked top K=10; kept relevance >0.70. Selected: {updates.get('entities', {}).get('chemical')}; accepted: {brief}; rejected top scores: {[(item.get('chemical_name'), item.get('relevance_score')) for item in rejected]}."
    if name == "structured_retrieval":
        return f"Applied exact SQL filters {updates.get('filters', {})}; counts: {updates.get('counts', {})}; aggregates: {list(updates.get('aggregate', {}))}; evidence rows returned: {len(updates.get('evidence', []))}."
    if name == "evidence_builder":
        ids = [(row.get("CDPHId"), row.get("ChemicalId")) for row in updates.get("evidence", [])[:5]]
        return f"Validated {len(updates.get('evidence', []))} evidence rows; sample (CDPHId, ChemicalId): {ids}."
    if name == "answer_synthesizer":
        usage = updates.get("model_usage", {})
        method = "local Ollama" if usage.get("status") == "completed" else "deterministic synthesis"
        return f"Synthesized with {method}; tokens: {usage.get('total_tokens', 0)}; local API cost: ${usage.get('api_cost_usd', 0):.2f}."
    if name == "output_guardrail":
        return f"Checked evidence identifiers and unsupported claims; warnings: {len(updates.get('warnings', []))}."
    return updates.get("answer", "Clarification or scope response returned.")


def _instrument(name: str, node):
    def run(state: AgentState) -> dict[str, Any]:
        started = perf_counter()
        updates = node(state)
        metric = {
            "step": name,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "details": _step_details(name, state, updates),
        }
        return {
            **updates,
            "step_metrics": [*state.get("step_metrics", []), metric],
        }

    return run


def _route_after_input(state: AgentState) -> str:
    return "out_of_scope" if state.get("out_of_scope") else "planner"


def _route_after_plan(state: AgentState) -> str:
    return str(state.get("retrieval_mode", "structured"))


@lru_cache(maxsize=1)
def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("input_guardrail", _instrument("input_guardrail", input_guardrail))
    graph.add_node("planner", _instrument("planner", planner_node))
    graph.add_node("entity_extraction", _instrument("entity_extraction", entity_extraction_node))
    graph.add_node("router", _instrument("router", route_node))
    graph.add_node("semantic_retrieval", _instrument("semantic_retrieval", semantic_retrieval_node))
    graph.add_node("structured_retrieval", _instrument("structured_retrieval", structured_retrieval_node))
    graph.add_node("evidence_builder", _instrument("evidence_builder", evidence_builder_node))
    graph.add_node("answer_synthesizer", _instrument("answer_synthesizer", answer_synthesizer_node))
    graph.add_node("clarification", _instrument("clarification", clarification_node))
    graph.add_node("out_of_scope", _instrument("out_of_scope", out_of_scope_node))
    graph.add_node("output_guardrail", _instrument("output_guardrail", output_guardrail_node))

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        _route_after_input,
        {"planner": "planner", "out_of_scope": "out_of_scope"},
    )
    graph.add_edge("planner", "entity_extraction")
    graph.add_edge("entity_extraction", "router")
    graph.add_conditional_edges(
        "router",
        _route_after_plan,
        {
            "structured": "structured_retrieval",
            "semantic": "semantic_retrieval",
            "clarify": "clarification",
            "out_of_scope": "out_of_scope",
        },
    )
    graph.add_conditional_edges(
        "semantic_retrieval",
        lambda state: "clarification" if state.get("needs_clarification") else "structured_retrieval",
        {"clarification": "clarification", "structured_retrieval": "structured_retrieval"},
    )
    graph.add_edge("structured_retrieval", "evidence_builder")
    graph.add_edge("evidence_builder", "answer_synthesizer")
    for node in ("answer_synthesizer", "clarification", "out_of_scope"):
        graph.add_edge(node, "output_guardrail")
    graph.add_edge("output_guardrail", END)
    return graph.compile()


def ask(
    question: str,
    limit: int = 20,
    use_local_model: bool = True,
    conversation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    result = build_workflow().invoke({
        "question": question,
        "conversation_context": conversation_context or {},
        "limit": limit,
        "use_local_model": use_local_model,
        "warnings": [],
        "trace": [],
        "step_metrics": [],
    })
    query_plan = result.get("query_plan", {})
    if result.get("filters"):
        query_plan["filters"] = result["filters"]
    if result.get("semantic_candidates"):
        query_plan["semantic_candidates"] = result["semantic_candidates"]
    if result.get("semantic_rejected"):
        query_plan["semantic_rejected"] = result["semantic_rejected"]
    return {
        "answer": result.get("answer", "No answer was produced."),
        "evidence": result.get("evidence", []),
        "query_plan": query_plan,
        "summary": {"counts": result.get("counts", {}), "aggregate": result.get("aggregate", {})},
        "confidence": result.get("confidence", "low"),
        "warnings": result.get("warnings", []),
        "trace": result.get("trace", []),
        "step_metrics": result.get("step_metrics", []),
        "model_usage": result.get("model_usage", {}),
    }