import re

from src.schemas.query_plan import QueryPlan
from src.tools.date_normalization import normalize_date_expression


DATE_PHRASES = (
    ("most recent report", "most_recent_reported"),
    ("discontinued", "discontinued"),
    ("removed", "removed"),
    ("reformulated", "removed"),
    ("reported", "reported"),
)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
SAFETY_TERMS = (
    "safe",
    "safety",
    "pregnancy",
    "pregnant",
    "side effect",
    "toxic",
    "harmful",
    "health effect",
    "cancer",
    "allergy",
    "safe for children",
    "breastfeeding",
)
GROUP_BY_PATTERNS = (
    ("company", r"\bcompanies\b|\bcompany\b|\bmanufacturers?\b"),
    ("chemical", r"\bchemicals?\b|\bingredients?\b"),
    ("category", r"\b(?:product\s+)?categories\b|\bcategory\b"),
    ("subcategory", r"\bsubcategories\b|\bsubcategory\b"),
    ("brand", r"\bbrands?\b"),
)


class PlannerAgent:
    def plan(self, question: str) -> QueryPlan:
        text = question.lower()
        group_by = next(
            (field for field, pattern in GROUP_BY_PATTERNS if re.search(pattern, text)),
            None,
        )
        top_match = re.search(r"\btop\s+(\d+)\b", text)
        top_n = int(top_match.group(1)) if top_match else None
        grouped_count = bool(
            group_by
            and re.search(r"\b(?:for each|per|by)\s+(?:the\s+)?(?:companies?|company|manufacturers?|chemicals?|ingredients?|categories|category|subcategories|subcategory|brands?)\b", text)
            and re.search(r"\b(?:number|count|how many)\b", text)
        )
        ranked_groups = bool(
            group_by
            and (top_n is not None or re.search(r"\b(?:most|highest|largest|fewest|least|lowest|ascending|descending)\b", text))
        )
        aggregation_requested = grouped_count or ranked_groups
        measure = (
            "count_ingredient_records"
            if aggregation_requested and re.search(r"\b(?:ingredient\s+)?records\b", text)
            else "count_distinct_products"
        )
        order_by = "ingredient_records" if measure == "count_ingredient_records" else "product_count"
        order_direction = "asc" if re.search(r"\b(?:fewest|least|lowest|ascending)\b", text) else "desc"
        if any(term in text for term in SAFETY_TERMS):
            intent = "safety_question"
        elif aggregation_requested:
            intent = "aggregation"
        elif re.search(r"\bhow many\b.*\bcompanies\b", text):
            intent = "company_count"
        elif re.search(r"\bhow many\b.*\bchemicals?\b", text):
            intent = "chemical_count"
        elif re.search(r"\bhow many\b.*\bproducts?\b", text):
            intent = "product_count"
        elif re.search(r"\b(?:which|what)\s+companies\b|\bcompanies\s+(?:that|reported)\b", text):
            intent = "company_aggregation" if "how many products" in text or "product count" in text else "company_lookup"
        elif any(term in text for term in ("data quality", "missing values", "dataset stats", "how many rows", "dataset size")):
            intent = "data_quality"
        elif any(term in text for term in ("compare", "comparison", "versus", " vs ")):
            intent = "compare"
        elif any(term in text for term in ("trend", "over time", "by year", "yearly")):
            intent = "trend"
        elif any(term in text for term in ("summarize", "summary", "overview")):
            intent = "summarize"
        elif any(term in text for term in ("list", "show", "which products")):
            intent = "list"
        else:
            intent = "lookup"

        count_requested = bool(re.search(r"\bhow many\b|\bcount\b", text))
        aggregation_target = None
        if re.search(r"\bcompanies?\b|\bmanufacturers?\b", text):
            aggregation_target = "companies"
        elif re.search(r"\bchemicals?\b|\bingredients?\b", text):
            aggregation_target = "chemicals"
        elif re.search(r"\bproducts?\b", text):
            aggregation_target = "products"
        output_requests: list[tuple[int, str]] = []
        explicit_product_count = bool(re.search(r"\b(?:how many|number of|count of)\s+(?:unique\s+)?products?\b", text))
        company_count_requested = bool(
            re.search(r"\b(?:how many|number of|count of)\s+(?:unique\s+)?companies?\b", text)
            or re.search(r"\bunique\s+companies\b", text)
        )
        company_list_requested = bool(re.search(r"\b(?:which|what)\s+companies\b|\bcompanies\s+that\b", text))
        if explicit_product_count:
            match = re.search(r"\b(?:how many|number of|count of)\s+(?:unique\s+)?products?\b", text)
            output_requests.append((match.start() if match else 0, "product_count"))
        if company_count_requested:
            match = re.search(r"\b(?:how many|number of|count of)\s+(?:unique\s+)?companies?\b|\bunique\s+companies\b", text)
            output_requests.append((match.start() if match else 0, "company_count"))
        if company_list_requested:
            match = re.search(r"\b(?:which|what)\s+companies\b|\bcompanies\s+that\b", text)
            output_requests.append((match.start() if match else 0, "company_list"))
        output_targets = [target for _position, target in sorted(output_requests)]
        if len(output_targets) > 1 and not aggregation_requested:
            intent = "multi_output"

        has_reported_date_language = bool(
            re.search(r"\breported\b", text)
            and re.search(r"\b(?:in|during|after|before|from|until|between|on)\b", text)
        )
        date_field = "reported" if has_reported_date_language else next((field for phrase, field in DATE_PHRASES if phrase in text), None)
        if intent == "trend" and date_field is None:
            date_field = "reported"
        normalized_date = normalize_date_expression(question, date_field)
        date_operator = normalized_date.operator
        date_from = normalized_date.date_from
        date_to = normalized_date.date_to
        if date_field == "discontinued" and date_operator is None:
            date_operator = "exists"
        warnings = []
        years = [int(value) for value in YEAR_PATTERN.findall(text)]
        if years and not date_field:
            warnings.append("A year was mentioned without a lifecycle date; no date filter was applied.")
        return QueryPlan(
            intent=intent,
            date_field=date_field,
            date_from=date_from,
            date_to=date_to,
            date_operator=date_operator,
            aggregation_target=aggregation_target,
            group_by=group_by if aggregation_requested else None,
            measure=measure if aggregation_requested else None,
            order_by=order_by if aggregation_requested else None,
            order_direction=order_direction,
            top_n=top_n,
            distinct=count_requested or aggregation_requested,
            count=count_requested or aggregation_requested,
            output_targets=output_targets,
            require_discontinued="discontinued" in text,
            retrieval_mode="structured",
            warnings=warnings,
        )