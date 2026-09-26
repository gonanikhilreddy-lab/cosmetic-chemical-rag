import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


class AnswerSynthesizerAgent:
    @staticmethod
    def _group_products(evidence: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in evidence:
            product_id = row.get("CDPHId")
            if product_id is not None:
                grouped.setdefault(product_id, []).append(row)
        return list(grouped.values())

    @staticmethod
    def _citation_for(rows: list[dict[str, Any]]) -> str:
        citations = []
        seen = set()
        for row in rows:
            key = (row.get("CDPHId"), row.get("CSFId"), row.get("ChemicalId"), row.get("CasNumber"))
            if key in seen:
                continue
            seen.add(key)
            fields = [f"CDPHId {key[0]}"]
            if key[1] is not None:
                fields.append(f"CSFId {key[1]}")
            if key[2] is not None:
                fields.append(f"ChemicalId {key[2]}")
            if key[3]:
                fields.append(f"CAS {key[3]}")
            citations.append(" · ".join(fields))
        return "; ".join(citations)

    @staticmethod
    def _display_text(value: Any) -> str:
        return " ".join(str(value or "").replace("\ufffd", "").split()).strip().rstrip(",")

    @staticmethod
    def _summary_line(state: dict[str, Any]) -> str:
        intent = state["intent"]
        counts = state.get("counts", {})
        aggregate = state.get("aggregate", {})
        records = counts.get("ingredient_records", 0)
        products = counts.get("product_count", 0)
        if intent == "data_quality":
            stats = aggregate["dataset"]
            return (
                f"The dataset contains {stats['ingredient_records']:,} ingredient records across "
                f"{stats['product_count']:,} products. Most-recent reporting spans "
                f"{stats['first_reported']} to {stats['latest_reported']}."
            )
        if intent == "trend":
            rows = aggregate.get("trend", [])
            if not rows:
                return "No reporting trend records matched the selected filters."
            return "; ".join(
                f"{row['year']}: {row['product_count']:,} products, {row['ingredient_records']:,} ingredient records"
                for row in rows
            )
        if intent in ("company_lookup", "company_count", "company_aggregation"):
            companies = aggregate.get("companies", [])
            company_count = aggregate.get("company_count", len(companies))
            return (
                f"Found {company_count:,} companies reporting {products:,} products "
                f"across {records:,} ingredient records."
            )
        if intent == "compare" and aggregate.get("comparison"):
            return "; ".join(
                f"{row['entity']}: {row['product_count']:,} products, {row['ingredient_records']:,} ingredient records"
                for row in aggregate["comparison"]
            )
        if records == 0:
            coverage = aggregate.get("date_coverage", {})
            date_field = state.get("date_field")
            date_from = state.get("date_from")
            entities = state.get("entities", {})
            if coverage.get("product_count") and date_field and date_from:
                year = date_from[:4]
                lifecycle = date_field.replace("_", " ").capitalize()
                entity_labels = {
                    "brand": "Brand",
                    "company": "Company",
                    "chemical": "Chemical",
                    "product": "Product",
                    "category": "Category",
                    "subcategory": "Subcategory",
                }
                scope = ", ".join(
                    f"{entity_labels.get(kind, kind)} {value}"
                    for kind, value in entities.items()
                )
                scope_text = f" for {scope}" if scope else ""
                return (
                    f"No products{scope_text} were recorded as {lifecycle.lower()} in {year}. "
                    f"The available matching records have {lifecycle.lower()} dates from "
                    f"{coverage['first_date']} through {coverage['latest_date']} "
                    f"({coverage['product_count']:,} products). This describes the dataset only."
                )
            return "No records matched in the available dataset. This does not prove that no such products exist elsewhere."
        date_field = state.get("date_field")
        date_from = state.get("date_from")
        date_to = state.get("date_to")
        date_operator = state.get("date_operator")
        if date_field and date_from:
            lifecycle = date_field.replace("_", " ").capitalize()
            if date_operator == "after":
                date_text = f"after {date_from}"
            elif date_operator == "before":
                date_text = f"before {date_from}"
            elif date_operator == "on":
                date_text = f"on {date_from}"
            elif date_operator == "through":
                date_text = f"through {date_from}"
            elif date_operator == "between" or date_operator == "after_before":
                date_text = f"between {date_from} and {date_to}"
            elif date_operator == "from":
                date_text = f"from {date_from}"
            else:
                date_text = f"in {date_from[:4]}"
            return f"Found {products:,} products with {lifecycle} {date_text} ({records:,} ingredient records)."
        return f"Found {products:,} products across {records:,} matching ingredient records."

    def _format_answer(self, summary_line: str, state: dict[str, Any]) -> str:
        lines = ["### Summary", summary_line]
        if state.get("result_type") == "companies":
            companies = state.get("aggregate", {}).get("companies", [])
            if companies:
                lines.extend(["", "### Reporting companies"])
                for company in companies:
                    lines.append(
                        f"- **{self._display_text(company.get('CompanyName'))}**: "
                        f"{company.get('product_count', 0):,} products, "
                        f"{company.get('ingredient_records', 0):,} ingredient records."
                    )
            return "\n".join(lines)
        comparisons = state.get("aggregate", {}).get("comparison", [])
        if state.get("intent") == "compare" and comparisons:
            lines.extend(["", "### Comparison results"])
            for group in comparisons:
                lines.append(
                    f"- **{group['entity']}**: {group['product_count']:,} products, "
                    f"{group['ingredient_records']:,} ingredient records."
                )
                rows = group.get("evidence", [])
                if rows:
                    for index, row in enumerate(rows[:5], start=1):
                        name = self._display_text(row.get("ProductName")) or "Unnamed product"
                        company = self._display_text(row.get("CompanyName"))
                        company_text = f" by {company}" if company else ""
                        lines.append(f"  {index}. {name}{company_text}")
                        lines.append(f"     - **Evidence:** {self._citation_for([row])}")
                else:
                    lines.append("  - No matching row examples for these shared filters.")
            return "\n".join(lines)

        groups = self._group_products(state.get("evidence", []))
        counts = state.get("counts", {})
        total_products = counts.get("product_count", len(groups))
        if groups:
            shown = groups[:10]
            lines.extend(["", f"### Matching products ({total_products:,} total; showing {len(shown)})"])
            for index, rows in enumerate(shown, start=1):
                first = rows[0]
                product_name = self._display_text(first.get("ProductName")) or "Unnamed product"
                company = self._display_text(first.get("CompanyName")) or "Company not listed"
                lines.append(f"{index}. **{product_name}**")
                details = [f"Company: {company}"]
                if first.get("BrandName"):
                    details.append(f"Brand: {self._display_text(first['BrandName'])}")
                if first.get("PrimaryCategory"):
                    details.append(f"Category: {self._display_text(first['PrimaryCategory'])}")
                if first.get("DiscontinuedDate"):
                    details.append(f"Discontinued: {self._display_text(first['DiscontinuedDate'])}")
                chemicals = []
                seen_chemicals = set()
                for row in rows:
                    chemical = row.get("ChemicalName")
                    cas_number = row.get("CasNumber")
                    chemical_key = (chemical, cas_number, row.get("ChemicalId"))
                    if chemical_key in seen_chemicals:
                        continue
                    seen_chemicals.add(chemical_key)
                    if chemical:
                        chemicals.append(self._display_text(chemical) + (f" (CAS {self._display_text(cas_number)})" if cas_number else ""))
                if chemicals:
                    details.append("Reported chemicals: " + ", ".join(chemicals[:5]))
                lines.append("   - " + " · ".join(details))
                lines.append(f"   - **Evidence:** {self._citation_for(rows)}")
            if total_products > len(shown):
                lines.append(f"\nShowing {len(shown)} products; narrow the filters or increase the evidence limit to inspect more.")
        elif state.get("intent") != "data_quality":
            lines.extend(["", "### Matching products", "No row-level product evidence was returned."])
        return "\n".join(lines)

    def deterministic(self, state: dict[str, Any]) -> str:
        return self._format_answer(self._summary_line(state), state)

    def synthesize(self, question: str, state: dict[str, Any], use_local_model: bool) -> tuple[str, list[str], dict[str, Any]]:
        fallback = self.deterministic(state)
        warnings: list[str] = []
        evidence = state.get("evidence", [])
        if not use_local_model or not evidence:
            status = "disabled" if not use_local_model else "skipped_no_evidence"
            return fallback, warnings, {
                "model": OLLAMA_MODEL,
                "provider": "local Ollama",
                "status": status,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_cost_usd": 0.0,
                "cost_note": "Local inference has no API usage charge.",
            }
        if state.get("intent") != "summarize":
            return fallback, warnings, {
                "model": OLLAMA_MODEL,
                "provider": "local Ollama",
                "status": "skipped_sql_grounded_answer",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_cost_usd": 0.0,
                "cost_note": "Structured products, counts, dates, and citations are rendered directly from SQL evidence.",
            }
        model = None
        usage: dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "provider": "local Ollama",
            "status": "fallback",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_cost_usd": 0.0,
            "cost_note": "Local inference has no API usage charge.",
        }
        try:
            model = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
            response = model.invoke(
                "Answer the question using only the supplied database evidence and counts. "
                "Do not infer toxicity, safety, causation, exposure, or facts outside the data. "
                "Never invent counts, dates, names, or identifiers. Be concise.\n\n"
                f"Question: {question}\n"
                f"Verified counts: {json.dumps(state.get('counts', {}), ensure_ascii=True)}\n"
                f"Aggregate: {json.dumps(state.get('aggregate', {}), ensure_ascii=True, default=str)}\n"
                f"Evidence rows: {json.dumps(evidence[:5], ensure_ascii=True, default=str)}\n"
                "Write only the natural-language answer; do not invent citations."
            )
            metadata = response.response_metadata
            usage.update({
                "status": "completed",
                "prompt_tokens": int(metadata.get("prompt_eval_count", 0)),
                "completion_tokens": int(metadata.get("eval_count", 0)),
                "total_tokens": int(metadata.get("prompt_eval_count", 0)) + int(metadata.get("eval_count", 0)),
                "total_duration_ms": round(metadata.get("total_duration", 0) / 1_000_000, 2),
                "load_duration_ms": round(metadata.get("load_duration", 0) / 1_000_000, 2),
                "prompt_duration_ms": round(metadata.get("prompt_eval_duration", 0) / 1_000_000, 2),
                "generation_duration_ms": round(metadata.get("eval_duration", 0) / 1_000_000, 2),
            })
            narrative = str(response.content).strip()
            if not narrative or len(narrative) > 1600:
                raise ValueError("Local model returned an empty or oversized answer.")
            if re.search(r"\b(safe|dangerous|toxic|harmful|risk|exposure)\b", narrative, re.IGNORECASE):
                raise ValueError("Local model returned an unsupported product-safety claim.")
            return self._format_answer(narrative, state), warnings, usage
        except Exception as error:
            warnings.append(f"Local answer model unavailable or rejected; used deterministic synthesis ({type(error).__name__}).")
            return fallback, warnings, usage
        finally:
            if model is not None:
                model._client.close()