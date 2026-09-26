import re
from functools import lru_cache

from rapidfuzz import fuzz, process

from src.agents.planner import YEAR_PATTERN
from src.schemas.query_plan import QueryPlan
from src.tools.structured_query import cas_values_for_chemical, entity_values


ENTITY_TYPES = (
    "company", "brand", "product", "chemical", "cas", "category", "subcategory",
)
CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:what\s+about|how\s+about|and\b|also\b|same\b|those\b|them\b|that\b|it\b|now\b|then\b|what\s+else\b|which\s+ones\b|which\s+of\s+(?:those|them|these)\b|how\s+many\s+of\s+(?:those|them|these)\b)",
    re.IGNORECASE,
)
MARKERS = {
    "company": ("company", "manufacturer", "from", "by"),
    "brand": ("brand",),
    "product": ("product name", "product called", "product named"),
    "chemical": ("chemical", "ingredient", "contain", "contains", "containing", "with"),
    "cas": ("cas",),
    "category": ("category",),
    "subcategory": ("subcategory",),
}


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.replace("\ufffd", " ").lower()).split())


def canonical_value(value: str) -> str:
    return " ".join(value.replace("\ufffd", " ").split()).strip()


@lru_cache(maxsize=None)
def _values(entity_type: str) -> tuple[str, ...]:
    return tuple(entity_values(entity_type))


class EntityExtractionAgent:
    def extract(
        self,
        question: str,
        plan: QueryPlan,
        conversation_context: dict | None = None,
    ) -> tuple[QueryPlan, str | None, bool]:
        normalized_question = f" {normalize(question)} "
        matches: dict[str, list[str]] = {}
        for kind in ENTITY_TYPES:
            found = [
                value for value in _values(kind)
                if len(normalize(value)) >= 3 and f" {normalize(value)} " in normalized_question
            ]
            unique: dict[str, str] = {}
            for value in found:
                unique.setdefault(normalize(value), canonical_value(value))
            if unique:
                matches[kind] = sorted(unique.values(), key=lambda value: len(normalize(value)), reverse=True)

        cas_match = CAS_PATTERN.search(question)
        if cas_match:
            matches["cas"] = [cas_match.group(0)]

        def explicit(kind: str) -> bool:
            return any(re.search(rf"\b{re.escape(marker)}\b", question, re.IGNORECASE) for marker in MARKERS[kind])

        comparison_match = re.search(
            r"\bcompare(?:\s+(brands?|companies|manufacturers|products|chemicals|categories|subcategories))?\b",
            question,
            re.IGNORECASE,
        )
        comparison_kind = None
        if comparison_match:
            comparison_label = (comparison_match.group(1) or "").lower()
            comparison_kind = {
                "brand": "brand",
                "brands": "brand",
                "companies": "company",
                "company": "company",
                "manufacturers": "company",
                "manufacturer": "company",
                "product": "product",
                "products": "product",
                "chemical": "chemical",
                "chemicals": "chemical",
                "categories": "category",
                "category": "category",
                "subcategories": "subcategory",
                "subcategory": "subcategory",
            }.get(comparison_label)
            comparison_tail_start = comparison_match.end()
            if comparison_label:
                comparison_tail_start = comparison_match.end()
            if comparison_kind:
                comparison_tail = question[comparison_tail_start:]
                boundary = re.search(
                    r"\b(?:for|in|within|where|discontinued|reported|containing|with)\b",
                    comparison_tail,
                    re.IGNORECASE,
                )
                compared_text = comparison_tail[:boundary.start()] if boundary else comparison_tail
                compared_normalized = f" {normalize(compared_text)} "
                compared_values = []
                for value in _values(comparison_kind):
                    normalized_value = normalize(value)
                    if normalized_value and f" {normalized_value} " in compared_normalized:
                        compared_values.append(value)
                canonical_values: dict[str, str] = {}
                for value in compared_values:
                    canonical_values.setdefault(normalize(value), value)
                if len(canonical_values) > 1:
                    matches[comparison_kind] = sorted(
                        canonical_values.values(), key=lambda value: len(normalize(value)), reverse=True
                    )
            elif plan.intent == "compare":
                comparison_tail = question[comparison_tail_start:]
                boundary = re.search(
                    r"\b(?:for|in|within|where|discontinued|reported|containing|with)\b",
                    comparison_tail,
                    re.IGNORECASE,
                )
                compared_text = comparison_tail[:boundary.start()] if boundary else comparison_tail
                compared_normalized = f" {normalize(compared_text)} "
                matching_kinds = []
                canonical_by_kind: dict[str, list[str]] = {}
                for candidate_kind in ("brand", "company", "product", "chemical", "category", "subcategory"):
                    found_values = []
                    for value in _values(candidate_kind):
                        normalized_value = normalize(value)
                        if normalized_value and f" {normalized_value} " in compared_normalized:
                            found_values.append(value)
                    canonical: dict[str, str] = {}
                    for value in found_values:
                        canonical.setdefault(normalize(value), value)
                    if len(canonical) > 1:
                        matching_kinds.append(candidate_kind)
                        canonical_by_kind[candidate_kind] = list(canonical.values())
                if len(matching_kinds) == 1:
                    comparison_kind = matching_kinds[0]
                    matches[comparison_kind] = sorted(
                        canonical_by_kind[comparison_kind],
                        key=lambda value: len(normalize(value)),
                        reverse=True,
                    )
                elif len(matching_kinds) > 1:
                    plan.warnings.append(
                        "Comparison entity type is ambiguous across: " + ", ".join(matching_kinds) + ". Specify brands, companies, products, or chemicals."
                    )

        # A shorter label found inside a fully matched longer category is not
        # another entity unless the question explicitly names that entity type.
        for kind, values in list(matches.items()):
            if explicit(kind):
                continue
            filtered = []
            for value in values:
                normalized_value = normalize(value)
                nested_in_other = any(
                    other_kind != kind
                    and any(
                        len(normalize(other_value)) > len(normalized_value)
                        and f" {normalized_value} " in f" {normalize(other_value)} "
                        for other_value in other_values
                    )
                    for other_kind, other_values in matches.items()
                )
                if not nested_in_other:
                    filtered.append(value)
            if filtered:
                matches[kind] = filtered
            elif kind in matches:
                matches.pop(kind)

        if "category" not in matches and re.search(r"\bmakeup products?\b", question, re.IGNORECASE):
            makeup_category = next(
                (value for value in _values("category") if normalize(value) == "makeup products non permanent"),
                None,
            )
            if makeup_category:
                matches["category"] = [makeup_category]

        if re.search(r"\bsubcategory\b", question, re.IGNORECASE):
            hint_match = re.search(r"\b([a-z][a-z -]{1,40}?)\s+subcategory\b", question, re.IGNORECASE)
            if hint_match:
                stop_words = {"in", "the", "a", "an", "for", "under", "within"}
                hint_words = [word for word in hint_match.group(1).strip().split() if word.lower() not in stop_words]
                hint_text = " ".join(hint_words[-3:])
                choices = process.extract(
                    hint_text,
                    _values("subcategory"),
                    scorer=fuzz.WRatio,
                    limit=3,
                    score_cutoff=60,
                )
                if choices and (len(choices) == 1 or choices[0][1] - choices[1][1] >= 8):
                    matches["subcategory"] = [choices[0][0]]
                    plan.warnings.append(
                        f"Resolved partial subcategory {hint_text!r} to {choices[0][0]!r}."
                    )
                    normalized_subcategory = normalize(choices[0][0])
                    for kind in ("product", "brand", "chemical"):
                        if not explicit(kind) and kind in matches:
                            remaining = [
                                value for value in matches[kind]
                                if not any(
                                    f" {variant} " in f" {normalized_subcategory} "
                                    for variant in (
                                        normalize(value),
                                        normalize(value) + "s",
                                        normalize(value).removesuffix("s"),
                                    )
                                )
                            ]
                            if remaining:
                                matches[kind] = remaining
                            else:
                                matches.pop(kind, None)

        if matches.get("company") and not explicit("brand"):
            normalized_company = normalize(matches["company"][0])
            matches["brand"] = [
                name for name in matches.get("brand", [])
                if normalize(name) not in normalized_company
            ]
            if not matches["brand"]:
                matches.pop("brand", None)
        if matches.get("chemical") and not explicit("product"):
            normalized_chemical = normalize(matches["chemical"][0])
            matches["product"] = [
                name for name in matches.get("product", [])
                if normalize(name) != normalized_chemical
            ]
            if not matches["product"]:
                matches.pop("product", None)
        if matches.get("chemical") and not explicit("brand"):
            normalized_chemical = normalize(matches["chemical"][0])
            matches["brand"] = [
                name for name in matches.get("brand", [])
                if normalize(name) != normalized_chemical
            ]
            if not matches["brand"]:
                matches.pop("brand", None)

        entities: dict[str, str] = {}
        comparisons: dict[str, list[str]] = {}
        for kind, values in matches.items():
            if plan.intent == "compare" and kind == comparison_kind and len(values) > 1:
                comparisons[kind] = values
            else:
                entities[kind] = values[0]
                if len(values) > 1:
                    plan.warnings.append(f"Several {kind} values matched; using {values[0]!r}.")

        prior_plan = (conversation_context or {}).get("query_plan", {})
        context_reference = None
        if re.search(r"\b(?:this|that)\s+(?:chemical|ingredient)|\b(?:this|that)\s+cas(?:\s+number)?\b|\bit\b", question, re.IGNORECASE):
            context_reference = "chemical"
        elif re.search(r"\b(?:those|these)\s+companies\b|\bsame\s+companies\b|\bwhich\s+companies\s+are\s+(?:they|those)\b", question, re.IGNORECASE):
            context_reference = "company"
        elif re.search(r"\b(?:those|these)\s+products\b|\b(?:which|how\s+many)\s+of\s+(?:those|them|these)\b|\bthem\b|\bsame\s+product\b", question, re.IGNORECASE):
            context_reference = "product_context"
        elif prior_plan and re.match(r"^\s*(?:what\s+about|how\s+about|same)\b", question, re.IGNORECASE):
            context_reference = "product_context"

        if prior_plan and context_reference:
            plan.context_reference = context_reference
            if re.search(r"\bhow\s+many\s+of\s+(?:those|them|these)\b", question, re.IGNORECASE):
                plan.intent = "product_count"
                plan.aggregation_target = "products"
                plan.distinct = True
                plan.count = True
            allowed_kinds = {
                "chemical": {"chemical", "cas"},
                "company": {"company", "chemical", "cas"},
                "product_context": {"company", "brand", "product", "category", "subcategory", "chemical", "cas"},
            }[context_reference]
            for kind, value in prior_plan.get("entities", {}).items():
                if kind not in allowed_kinds:
                    continue
                if kind not in entities and kind not in comparisons:
                    entities[kind] = value
                    plan.inherited_entities[kind] = value

            if plan.inherited_entities:
                plan.warnings.append(
                    f"Follow-up context applied from the explicit {context_reference} reference; inherited fields: {', '.join(plan.inherited_entities)}."
                )

        ambiguous = []
        comparison_names_entity_type = comparison_kind in ("brand", "company")
        if not comparison_names_entity_type and not explicit("brand") and not explicit("company") and matches.get("brand"):
            selected_brand = normalize(matches["brand"][0])
            possible_companies = [
                value for value in _values("company")
                if f" {selected_brand} " in f" {normalize(value)} "
            ]
            if possible_companies:
                ambiguous = [f"brand: {matches['brand'][0]}"] + [f"company: {value}" for value in possible_companies[:3]]

        chemical_hint = None
        if "chemical" not in entities and "cas" not in entities:
            match = re.search(
                r"\b(?:contain(?:s|ing)?|with|chemical(?:\s+name)?|ingredient(?:\s+name)?)\s+"
                r"(?:(?:the|called|named)\s+)?(.+?)"
                r"(?=\s+(?:and|or)\s+(?:were|was|in|from|under|by)\b|[?.!,;]|$)",
                question,
                re.IGNORECASE,
            )
            if match:
                chemical_hint = match.group(1).strip(" \"'")
                chemical_hint = re.sub(
                    r"^(?:something similar to|something like|similar to|a chemical like)\s+",
                    "",
                    chemical_hint,
                    flags=re.IGNORECASE,
                )
                chemical_hint = re.sub(r"\s+(?:in|for|from)\s+.+$", "", chemical_hint, flags=re.IGNORECASE).strip()
                if re.match(
                    r"^(?:brand|company|manufacturer|product|category|subcategory|cas(?:\s+number)?)\b",
                    chemical_hint,
                    flags=re.IGNORECASE,
                ):
                    chemical_hint = None
                elif any(
                    normalize(chemical_hint) == normalize(value)
                    for values in matches.values()
                    for value in values
                ):
                    chemical_hint = None

        plan.entities = entities
        plan.context_reference = context_reference
        if "chemical" in entities and "cas" not in entities:
            chemical_cas = cas_values_for_chemical(entities["chemical"])
            if len(chemical_cas) == 1:
                entities["cas"] = chemical_cas[0]
        plan.comparisons = comparisons
        if ambiguous:
            plan.warnings.append("The name can refer to multiple entity types: " + "; ".join(ambiguous))
            plan.retrieval_mode = "clarify"
        elif chemical_hint:
            plan.retrieval_mode = "semantic"
        else:
            plan.retrieval_mode = "structured"
        if chemical_hint and not matches.get("chemical"):
            return plan, chemical_hint, False
        return plan, None, bool(ambiguous)