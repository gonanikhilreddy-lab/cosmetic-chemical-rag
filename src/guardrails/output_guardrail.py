from src.guardrails.grounding import evidence_is_valid


def validate_output(evidence: list[dict], answer: str) -> tuple[str, list[str]]:
    warnings = []
    if evidence and not evidence_is_valid(evidence):
        return "I could not verify the retrieved records, so I am withholding a factual answer.", [
            "Evidence validation failed: one or more rows were missing product or chemical identifiers."
        ]
    unsupported_safety_claim = any(
        term in answer.lower() for term in ("is safe", "is dangerous", "is toxic", "causes cancer", "causes birth defects")
    )
    if unsupported_safety_claim:
        return "The dataset reports chemical disclosures; it does not establish product safety or health effects.", [
            "Unsupported safety claim was blocked by the output guardrail."
        ]
    return answer, warnings