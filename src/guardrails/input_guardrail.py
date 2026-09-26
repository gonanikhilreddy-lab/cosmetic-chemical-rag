import re

from src.tools.structured_query import entity_values

DATASET_TERMS = re.compile(
    r"cosmetic|product|chemical|ingredient|brand|company|manufacturer|cas\b|category|subcategory|"
    r"reported|reporting|discontinued|reformulat|removed|dataset|data quality|how many rows",
    re.IGNORECASE,
)


def is_in_scope(question: str) -> bool:
    if DATASET_TERMS.search(question):
        return True
    normalized_question = " ".join(re.sub(r"[^a-z0-9]+", " ", question.lower()).split())
    return any(
        f" {' '.join(value.lower().split())} " in f" {normalized_question} "
        for entity_type in ("company", "brand", "chemical", "cas")
        for value in entity_values(entity_type)
        if len(value.strip()) >= 3
    )