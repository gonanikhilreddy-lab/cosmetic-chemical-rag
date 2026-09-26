from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    conversation_context: dict[str, Any]
    limit: int
    use_local_model: bool
    intent: str
    query_plan: dict[str, Any]
    entities: dict[str, str]
    filters: dict[str, Any]
    comparisons: dict[str, list[str]]
    date_field: str | None
    date_from: str | None
    date_to: str | None
    date_operator: str | None
    chemical_hint: str | None
    semantic_candidates: list[dict[str, Any]]
    semantic_rejected: list[dict[str, Any]]
    retrieval_mode: str
    needs_clarification: bool
    out_of_scope: bool
    counts: dict[str, int]
    aggregate: dict[str, Any]
    evidence: list[dict[str, Any]]
    warnings: list[str]
    trace: list[str]
    step_metrics: list[dict[str, Any]]
    model_usage: dict[str, Any]
    answer: str
    confidence: str
    result_type: str
    sql_date_predicate: str | None