from typing import Any


def evidence_is_valid(evidence: list[dict[str, Any]]) -> bool:
    return all(row.get("CDPHId") is not None and row.get("ChemicalId") is not None for row in evidence)