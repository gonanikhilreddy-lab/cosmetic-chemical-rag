from typing import Any

from src.agents.answer_synthesizer import AnswerSynthesizerAgent
from src.agents.entity_extractor import EntityExtractionAgent
from src.agents.planner import PlannerAgent
from src.agents.semantic_retrieval_agent import SemanticRetrievalAgent
from src.agents.structured_query_agent import StructuredQueryAgent
from src.guardrails.input_guardrail import is_in_scope
from src.guardrails.output_guardrail import validate_output
from src.schemas.agent_state import AgentState
from src.schemas.evidence import EvidenceRecord
from src.schemas.query_plan import QueryPlan


planner = PlannerAgent()
entity_extractor = EntityExtractionAgent()
semantic_retriever = SemanticRetrievalAgent()
structured_retriever = StructuredQueryAgent()
answer_synthesizer = AnswerSynthesizerAgent()


def input_guardrail(state: AgentState) -> dict[str, Any]:
    in_scope = is_in_scope(state["question"])
    trace = state.get("trace", []) + ["input guardrail: " + ("accepted dataset-related question" if in_scope else "rejected out-of-scope question")]
    return {"out_of_scope": not in_scope, "trace": trace}


def planner_node(state: AgentState) -> dict[str, Any]:
    plan = planner.plan(state["question"])
    return {
        "query_plan": plan.model_dump(),
        "intent": plan.intent,
        "date_field": plan.date_field,
        "date_from": plan.date_from,
        "date_to": plan.date_to,
        "warnings": plan.warnings,
        "trace": state.get("trace", []) + [f"planner: intent={plan.intent}"],
    }


def entity_extraction_node(state: AgentState) -> dict[str, Any]:
    plan = QueryPlan.model_validate(state["query_plan"])
    plan, hint, needs_clarification = entity_extractor.extract(
        state["question"], plan, conversation_context=state.get("conversation_context")
    )
    return {
        "query_plan": plan.model_dump(),
        "entities": plan.entities,
        "comparisons": plan.comparisons,
        "chemical_hint": hint,
        "needs_clarification": needs_clarification,
        "warnings": plan.warnings,
        "trace": state.get("trace", []) + ["entity extraction: resolved exact names, CAS, categories, and date constraints"],
    }


def route_node(state: AgentState) -> dict[str, Any]:
    plan = QueryPlan.model_validate(state["query_plan"])
    if state.get("chemical_hint"):
        plan.retrieval_mode = "semantic"
    route = "out_of_scope" if state.get("out_of_scope") else (
        "clarify" if state.get("needs_clarification") else
        "semantic" if state.get("chemical_hint") else "structured"
    )
    plan.retrieval_mode = {"semantic": "semantic", "structured": "structured", "clarify": "clarify", "out_of_scope": "out_of_scope"}[route]
    return {
        "query_plan": plan.model_dump(),
        "retrieval_mode": plan.retrieval_mode,
        "trace": state.get("trace", []) + [f"router: selected {route} retrieval"],
    }


def semantic_retrieval_node(state: AgentState) -> dict[str, Any]:
    candidates, selected, needs_clarification, rejected = semantic_retriever.search(state["chemical_hint"])
    warnings = list(state.get("warnings", []))
    plan = QueryPlan.model_validate(state["query_plan"])
    if candidates and candidates[0].get("retrieval") == "fuzzy-fallback":
        warnings.append("Vector search was unavailable; local fuzzy matching supplied candidate names.")
    elif candidates and candidates[0].get("vector_error"):
        warnings.append("Vector search was unavailable; BM25 candidates were used without dense retrieval.")
    if selected:
        plan.entities = {**plan.entities, "chemical": selected}
        plan.retrieval_mode = "hybrid"
        warnings.append(f"Hybrid vector/BM25 candidate {selected!r} passed the >0.70 relevance threshold and was verified against exact DuckDB rows.")
        trace_message = "semantic retrieval: fused vector/BM25 rankings, reranked, applied strict >0.70 threshold, then selected for SQL verification"
    else:
        plan.retrieval_mode = "clarify"
        warnings.append("No unique candidate passed the strict >0.70 relevance threshold; no SQL lookup was run.")
        trace_message = "semantic retrieval: rejected low-relevance candidates and stopped for clarification"
    return {
        "semantic_candidates": candidates,
        "semantic_rejected": rejected,
        "entities": plan.entities,
        "query_plan": plan.model_dump(),
        "needs_clarification": needs_clarification,
        "warnings": warnings,
        "trace": state.get("trace", []) + [trace_message],
    }


def structured_retrieval_node(state: AgentState) -> dict[str, Any]:
    results = structured_retriever.retrieve(
        question=state["question"],
        intent=state["intent"],
        entities=state.get("entities", {}),
        comparisons=state.get("comparisons", {}),
        date_field=state.get("date_field"),
        date_from=state.get("date_from"),
        date_to=state.get("date_to"),
        limit=state.get("limit", 20),
    )
    return {
        **results,
        "trace": state.get("trace", []) + ["structured retrieval: parameterized DuckDB filters returned records and aggregates"],
    }


def evidence_builder_node(state: AgentState) -> dict[str, Any]:
    verified = []
    warnings = list(state.get("warnings", []))
    for item in state.get("evidence", []):
        try:
            verified.append(EvidenceRecord.model_validate(item).model_dump(mode="json"))
        except Exception:
            warnings.append("A retrieved row failed evidence validation and was withheld.")
    return {
        "evidence": verified,
        "warnings": warnings,
        "trace": state.get("trace", []) + [f"evidence builder: validated {len(verified)} rows with CDPHId/ChemicalId"],
    }


def answer_synthesizer_node(state: AgentState) -> dict[str, Any]:
    response, model_warnings, model_usage = answer_synthesizer.synthesize(
        state["question"], state, use_local_model=state.get("use_local_model", True)
    )
    warnings = list(state.get("warnings", [])) + model_warnings
    if any(term in state["question"].lower() for term in ("safe", "toxic", "dangerous", "harmful", "risk", "exposure")):
        warnings.append("The disclosure dataset does not determine product safety, exposure, or individual health risk.")
    return {
        "answer": response,
        "model_usage": model_usage,
        "warnings": warnings,
        "confidence": "medium" if state.get("retrieval_mode") == "hybrid" else "high",
        "trace": state.get("trace", []) + ["answer synthesizer: composed the answer from retrieved evidence and local model"],
    }


def clarification_node(state: AgentState) -> dict[str, Any]:
    candidates = state.get("semantic_candidates", [])
    if candidates:
        choices = ", ".join(str(item["chemical_name"]) for item in candidates[:5])
        answer = f"I couldn't safely select one chemical match. Which did you mean: {choices}?"
    elif state.get("semantic_rejected"):
        best = state["semantic_rejected"][0]
        answer = (
            "No chemical candidate met the >0.70 relevance cutoff. "
            f"The closest was {best['chemical_name']} (score {best['relevance_score']:.3f}); "
            "please provide the exact chemical name or CAS number."
        )
    else:
        answer = "I found a name that can refer to more than one dataset entity. Did you mean the company or the brand?"
    return {
        "answer": answer,
        "confidence": "needs_clarification",
        "trace": state.get("trace", []) + ["ambiguity handler: paused retrieval for user clarification"],
    }


def out_of_scope_node(state: AgentState) -> dict[str, Any]:
    return {
        "answer": "This assistant answers questions about California cosmetic chemical disclosures. Please ask about a product, company, brand, chemical, CAS number, category, or reporting date.",
        "confidence": "low",
        "trace": state.get("trace", []) + ["input guardrail: returned out-of-scope guidance"],
    }


def output_guardrail_node(state: AgentState) -> dict[str, Any]:
    answer, warnings = validate_output(state.get("evidence", []), state.get("answer", ""))
    return {
        "answer": answer,
        "warnings": list(state.get("warnings", [])) + warnings,
        "trace": state.get("trace", []) + ["output guardrail: checked source IDs and blocked unsupported safety claims"],
    }