from src.schemas.query_plan import QueryPlan


def route_plan(plan: QueryPlan, needs_clarification: bool, out_of_scope: bool) -> str:
    if out_of_scope:
        return "out_of_scope"
    if needs_clarification or plan.retrieval_mode == "clarify":
        return "clarify"
    if plan.retrieval_mode == "semantic":
        return "semantic"
    return "structured"