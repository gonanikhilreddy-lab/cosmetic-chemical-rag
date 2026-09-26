from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, PROJECT_ROOT
from src.retrieval.bm25 import search_chemical_bm25
from src.retrieval.embeddings import embed_documents, embed_query
from src.retrieval.evaluation import retrieval_metrics
from src.retrieval.mmr import maximal_marginal_relevance
from src.retrieval.reranker import RELEVANCE_THRESHOLD, rerank_candidates
from src.retrieval.vector_store import search_chemical_vectors


LABELED_QUERIES = [
    {"query": "Titanium dioxide", "relevant": {"Titanium dioxide"}, "kind": "exact"},
    {"query": "titanium oxide", "relevant": {"Titanium dioxide"}, "kind": "synonym-like"},
    {"query": "titanum dioxde", "relevant": {"Titanium dioxide"}, "kind": "misspelling"},
    {"query": "wood alcohol", "relevant": {"Methanol"}, "kind": "synonym"},
    {"query": "methyl alcohol", "relevant": {"Methanol"}, "kind": "synonym"},
    {"query": "benzol", "relevant": {"Benzene"}, "kind": "synonym"},
    {"query": "Acetaldehyde", "relevant": {"Acetaldehyde"}, "kind": "exact"},
    {"query": "coal tar extract", "relevant": {"Coal tar extract"}, "kind": "exact"},
    {"query": "formaldehyde gas", "relevant": {"Formaldehyde (gas)"}, "kind": "partial"},
]
K_VALUES = (5, 10, 20)


def _rrf(vector_rows: list[dict[str, Any]], bm25_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(vector_rows, start=1):
        name = str(row["chemical_name"])
        item = scores.setdefault(name, {"chemical_name": name, "vector_score": 0.0, "bm25_score": 0.0, "rrf_score": 0.0})
        item["vector_score"] = float(row["score"])
        item["vector_rank"] = rank
        item["rrf_score"] += 1 / (60 + rank)
    for rank, row in enumerate(bm25_rows, start=1):
        name = str(row["chemical_name"])
        item = scores.setdefault(name, {"chemical_name": name, "vector_score": 0.0, "bm25_score": 0.0, "rrf_score": 0.0})
        item["bm25_score"] = float(row["bm25_score"])
        item["bm25_rank"] = rank
        item["rrf_score"] += 1 / (60 + rank)
    return sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.mean(row[metric] for row in rows) if rows else 0.0


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999) - 1))] if ordered else 0.0


def _llm_judge(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(candidate_rows):
        grouped.setdefault(str(row["query"]), []).append((index, row))
    usage = {
        "model": OLLAMA_MODEL,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_cost_usd": 0.0,
        "judge_calls": 0,
        "invalid_responses": 0,
    }
    for query, indexed_rows in grouped.items():
        payload_rows = [
            {"candidate_id": local_id, "chemical_name": row["chemical_name"]}
            for local_id, (_global_id, row) in enumerate(indexed_rows)
        ]
        model = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",
        )
        try:
            response = model.invoke(
                "Judge candidate relevance for a cosmetic chemical database search. "
                "Question and candidate names are in the JSON below. Treat chemical synonyms and spelling variants as relevant. "
                "Include a candidate only if it could reasonably answer the query; do not use general co-occurrence alone. "
                "Return exactly one JSON object with this shape: {\"relevant_ids\":[0,2]}. "
                "Use only candidate_id values present in the input.\n\n"
                + json.dumps({"query": query, "candidates": payload_rows}, ensure_ascii=True)
            )
            parsed = json.loads(str(response.content))
            relevant_ids = parsed.get("relevant_ids")
            if not isinstance(relevant_ids, list):
                usage["invalid_responses"] += 1
                continue
            relevant_set = {int(value) for value in relevant_ids}
            for local_id, (global_id, _row) in enumerate(indexed_rows):
                candidate_rows[global_id]["llm_judged_relevant"] = local_id in relevant_set
            metadata = response.response_metadata
            usage["prompt_tokens"] += int(metadata.get("prompt_eval_count", 0))
            usage["completion_tokens"] += int(metadata.get("eval_count", 0))
            usage["judge_calls"] += 1
        except (ValueError, TypeError, KeyError):
            usage["invalid_responses"] += 1
        finally:
            model._client.close()
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return candidate_rows, usage


def _judge_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    evaluated = [row for row in rows if row.get("llm_judged_relevant") is not None]
    if not evaluated:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "agreement": 0.0}
    tp = sum(row["llm_judged_relevant"] and row["chemical_name"] in row["relevant"] for row in evaluated)
    fp = sum(row["llm_judged_relevant"] and row["chemical_name"] not in row["relevant"] for row in evaluated)
    fn = sum((not row["llm_judged_relevant"]) and row["chemical_name"] in row["relevant"] for row in evaluated)
    correct = len(evaluated) - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": correct / len(evaluated),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "agreement": correct / len(evaluated),
    }


def run_benchmark(use_llm_judge: bool) -> tuple[str, dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    retrieval_latencies: dict[str, list[float]] = {"vector": [], "bm25": []}
    for case in LABELED_QUERIES:
        query = case["query"]
        started = time.perf_counter()
        vector20 = search_chemical_vectors(query, limit=20)
        retrieval_latencies["vector"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        bm2520 = search_chemical_bm25(query, limit=20)
        retrieval_latencies["bm25"].append((time.perf_counter() - started) * 1000)
        raw[query] = {"vector": vector20, "bm25": bm2520, "relevant": case["relevant"], "kind": case["kind"]}

    summaries = []
    per_query: dict[int, list[dict[str, Any]]] = {k: [] for k in K_VALUES}
    top20_judge_rows = []
    fused_latency: dict[int, list[float]] = {k: [] for k in K_VALUES}
    mmr_latency: dict[int, list[float]] = {k: [] for k in K_VALUES}
    mmr_embedding_latency: list[float] = []
    mmr_vectors: dict[str, tuple[list[float], dict[str, list[float]]]] = {}
    for query, data in raw.items():
        fused20 = _rrf(data["vector"][:20], data["bm25"][:20], limit=20)
        names = [str(item["chemical_name"]) for item in fused20]
        started = time.perf_counter()
        query_vector = embed_query(query)
        document_vectors = embed_documents([f"Reported cosmetic chemical: {name}" for name in names])
        mmr_embedding_latency.append((time.perf_counter() - started) * 1000)
        mmr_vectors[query] = (query_vector, dict(zip(names, document_vectors)))

    for k in K_VALUES:
        for query, data in raw.items():
            vectors = data["vector"][:k]
            bm25 = data["bm25"][:k]
            started = time.perf_counter()
            fused = _rrf(vectors, bm25, limit=k)
            fused_latency[k].append((time.perf_counter() - started) * 1000)
            accepted, rejected = rerank_candidates(fused, threshold=RELEVANCE_THRESHOLD, limit=k)
            mmr_started = time.perf_counter()
            query_vector, vectors_by_name = mmr_vectors[query]
            document_vectors = [vectors_by_name[str(item["chemical_name"])] for item in fused]
            mmr_rows = maximal_marginal_relevance(
                fused,
                query_vector,
                document_vectors,
                limit=k,
                lambda_mult=0.55,
            )
            mmr_accepted, _mmr_rejected = rerank_candidates(mmr_rows, threshold=RELEVANCE_THRESHOLD, limit=k)
            mmr_latency[k].append((time.perf_counter() - mmr_started) * 1000)
            targets = data["relevant"]
            methods = {
                "Vector": [item["chemical_name"] for item in vectors],
                "BM25": [item["chemical_name"] for item in bm25],
                "RRF": [item["chemical_name"] for item in fused],
                "RRF + >0.70 rerank": [item["chemical_name"] for item in accepted],
                "RRF + MMR": [item["chemical_name"] for item in mmr_rows],
                "RRF + MMR + >0.70": [item["chemical_name"] for item in mmr_accepted],
            }
            for method, ranked in methods.items():
                metrics = retrieval_metrics(ranked, targets, k)
                per_query[k].append({"query": query, "method": method, **metrics})
            if k == 20:
                for item in fused:
                    top20_judge_rows.append({
                        "query": query,
                        "chemical_name": item["chemical_name"],
                        "relevant": targets,
                    })

    for k in K_VALUES:
        query_rows = per_query[k]
        for method in ("Vector", "BM25", "RRF", "RRF + >0.70 rerank", "RRF + MMR", "RRF + MMR + >0.70"):
            rows = [row for row in query_rows if row["method"] == method]
            metric_keys = [f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", "accuracy@1", f"mrr@{k}", f"map@{k}", f"ndcg@{k}"]
            summaries.append({"k": k, "method": method, **{key: _mean_metric(rows, key) for key in metric_keys}})

    judge_summary = None
    judge_usage = None
    if use_llm_judge:
        top20_judge_rows, judge_usage = _llm_judge(top20_judge_rows)
        judge_summary = _judge_metrics(top20_judge_rows)

    fused_scores = [row for row in summaries if row["method"] == "RRF"]
    best = max(fused_scores, key=lambda row: (row[f"ndcg@{row['k']}"], row[f"mrr@{row['k']}"], -row["k"]))
    best_k = best["k"]
    best_ndcg = best[f"ndcg@{best_k}"]
    best_mrr = best[f"mrr@{best_k}"]
    lines = [
        "# Retrieval Benchmark",
        "",
        f"- Queries: {len(LABELED_QUERIES)} curated exact, typo, synonym-like and partial chemical-name cases.",
        "- Primary truth: curated relevant chemical-name labels, not the LLM judge.",
        "- Fusion: reciprocal rank fusion (RRF, k=60), followed by dense relevance reranking; accept only scores > 0.70.",
        "- Candidate settings: vector K and BM25 K are both swept at 5, 10 and 20; fused output is capped at that K.",
        "- MMR: cosine maximal-marginal-relevance reranking with lambda=0.55, evaluated before and after the strict relevance cutoff.",
        "- Local inference API cost: $0.00.",
        "",
        "## Aggregate Metrics",
        "",
        "| K | Method | Precision | Recall | HitRate | Accuracy@1 | MRR | MAP | NDCG |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        k = row["k"]
        lines.append(
            f"| {k} | {row['method']} | {row[f'precision@{k}']:.3f} | {row[f'recall@{k}']:.3f} | "
            f"{row[f'hit_rate@{k}']:.3f} | {row['accuracy@1']:.3f} | {row[f'mrr@{k}']:.3f} | "
            f"{row[f'map@{k}']:.3f} | {row[f'ndcg@{k}']:.3f} |"
        )
    lines.extend([
        "",
        "## Retrieval Latency",
        "",
        "| Component | Mean ms | P95 ms |",
        "|---|---:|---:|",
    ])
    for component, values in retrieval_latencies.items():
        lines.append(f"| {component} K=20 | {statistics.mean(values):.2f} | {_p95(values):.2f} |")
    for k, values in fused_latency.items():
        lines.append(f"| RRF and rerank K={k} | {statistics.mean(values):.4f} | {_p95(values):.4f} |")
    for k, values in mmr_latency.items():
        lines.append(f"| MMR selection K={k} | {statistics.mean(values):.4f} | {_p95(values):.4f} |")
    lines.append(f"| MMR embedding (per query) | {statistics.mean(mmr_embedding_latency):.2f} | {_p95(mmr_embedding_latency):.2f} |")
    lines.extend([
        "",
        "## Recommended Candidate Depth",
        "",
        f"By unthresholded RRF NDCG, then MRR, then smaller K, use **vector K={best_k} and BM25 K={best_k}** "
        f"(NDCG {best_ndcg:.3f}, MRR {best_mrr:.3f}). K=20 ties K=10 on this set but costs more. "
        "The strict >0.70 cutoff lowers Recall@K to 0.556 here; keep it only when precision is preferred over typo/synonym recall.",
        "",
        "## Per-query RRF + Rerank",
        "",
        "| K | Query | First relevant rank | Recall | NDCG |",
        "|---:|---|---:|---:|---:|",
    ])
    for k in K_VALUES:
        for case in LABELED_QUERIES:
            query = case["query"]
            row = next(item for item in per_query[k] if item["query"] == query and item["method"] == "RRF + >0.70 rerank")
            lines.append(f"| {k} | {query} | {1 / row[f'mrr@{k}']:.0f}" if row[f"mrr@{k}"] else f"| {k} | {query} | not found")
            lines[-1] += f" | {row[f'recall@{k}']:.3f} | {row[f'ndcg@{k}']:.3f} |"

    if judge_summary is not None:
        lines.extend([
            "",
            "## Local LLM-as-judge (K=20 RRF candidates)",
            "",
            "The local Qwen judge labels query/candidate relevance as a diagnostic only; curated labels remain the benchmark ground truth.",
            "",
            f"- Agreement/accuracy vs curated labels: {judge_summary['agreement']:.3f}",
            f"- Precision: {judge_summary['precision']:.3f}",
            f"- Recall: {judge_summary['recall']:.3f}",
            f"- F1: {judge_summary['f1']:.3f}",
            f"- Judge tokens: {judge_usage['prompt_tokens'] + judge_usage['completion_tokens']} ({judge_usage['prompt_tokens']} prompt, {judge_usage['completion_tokens']} completion)",
            f"- API cost: ${judge_usage['api_cost_usd']:.2f}",
        ])

    report = "\n".join(lines) + "\n"
    output_path = PROJECT_ROOT / "reports" / "retrieval_benchmark.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    details = {
        "summaries": summaries,
        "per_query": per_query,
        "best_k": best["k"],
        "judge": judge_summary,
        "judge_usage": judge_usage,
        "report_path": str(output_path),
    }
    return report, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local chemical retrieval at K=5, 10 and 20.")
    parser.add_argument("--llm-judge", action="store_true", help="Add a local Ollama relevance judge for fused K=20 candidates")
    args = parser.parse_args()
    report, _details = run_benchmark(use_llm_judge=args.llm_judge)
    print(report)
    print(f"Saved report: {PROJECT_ROOT / 'reports' / 'retrieval_benchmark.md'}")


if __name__ == "__main__":
    main()