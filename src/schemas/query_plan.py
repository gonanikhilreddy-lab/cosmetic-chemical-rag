from typing import Literal

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal["lookup", "list", "product_count", "chemical_count", "multi_output", "compare", "summarize", "trend", "data_quality", "safety_question", "company_lookup", "company_count", "company_aggregation"] = "lookup"
    entities: dict[str, str] = Field(default_factory=dict)
    inherited_entities: dict[str, str] = Field(default_factory=dict)
    comparisons: dict[str, list[str]] = Field(default_factory=dict)
    date_field: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    date_operator: Literal["exists", "after", "before", "on", "from", "through", "between", "range", "after_before"] | None = None
    aggregation_target: Literal["products", "companies", "chemicals"] | None = None
    distinct: bool = False
    count: bool = False
    output_targets: list[str] = Field(default_factory=list)
    context_reference: str | None = None
    require_discontinued: bool = False
    retrieval_mode: Literal["structured", "semantic", "hybrid", "clarify", "out_of_scope"] = "structured"
    inherited_date_constraint: bool = False
    warnings: list[str] = Field(default_factory=list)