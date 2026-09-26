# Retrieval Benchmark

- Queries: 9 curated exact, typo, synonym-like and partial chemical-name cases.
- Primary truth: curated relevant chemical-name labels, not the LLM judge.
- Fusion: reciprocal rank fusion (RRF, k=60), followed by dense relevance reranking; accept only scores > 0.70.
- Candidate settings: vector K and BM25 K are both swept at 5, 10 and 20; fused output is capped at that K.
- MMR: cosine maximal-marginal-relevance reranking with lambda=0.55, evaluated before and after the strict relevance cutoff.
- Local inference API cost: $0.00.

## Aggregate Metrics

| K | Method | Precision | Recall | HitRate | Accuracy@1 | MRR | MAP | NDCG |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | Vector | 0.200 | 1.000 | 1.000 | 0.667 | 0.754 | 0.754 | 0.813 |
| 5 | BM25 | 0.111 | 0.556 | 0.556 | 0.444 | 0.481 | 0.481 | 0.500 |
| 5 | RRF | 0.178 | 0.889 | 0.889 | 0.667 | 0.731 | 0.731 | 0.770 |
| 5 | RRF + >0.70 rerank | 0.111 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |
| 5 | RRF + MMR | 0.178 | 0.889 | 0.889 | 0.667 | 0.731 | 0.731 | 0.770 |
| 5 | RRF + MMR + >0.70 | 0.111 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |
| 10 | Vector | 0.100 | 1.000 | 1.000 | 0.667 | 0.754 | 0.754 | 0.813 |
| 10 | BM25 | 0.056 | 0.556 | 0.556 | 0.444 | 0.481 | 0.481 | 0.500 |
| 10 | RRF | 0.100 | 1.000 | 1.000 | 0.667 | 0.750 | 0.750 | 0.810 |
| 10 | RRF + >0.70 rerank | 0.056 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |
| 10 | RRF + MMR | 0.100 | 1.000 | 1.000 | 0.667 | 0.739 | 0.739 | 0.800 |
| 10 | RRF + MMR + >0.70 | 0.056 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |
| 20 | Vector | 0.050 | 1.000 | 1.000 | 0.667 | 0.754 | 0.754 | 0.813 |
| 20 | BM25 | 0.028 | 0.556 | 0.556 | 0.444 | 0.481 | 0.481 | 0.500 |
| 20 | RRF | 0.050 | 1.000 | 1.000 | 0.667 | 0.750 | 0.750 | 0.810 |
| 20 | RRF + >0.70 rerank | 0.028 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |
| 20 | RRF + MMR | 0.050 | 1.000 | 1.000 | 0.667 | 0.729 | 0.729 | 0.791 |
| 20 | RRF + MMR + >0.70 | 0.028 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 |

## Retrieval Latency

| Component | Mean ms | P95 ms |
|---|---:|---:|
| vector K=20 | 2537.73 | 2618.99 |
| bm25 K=20 | 7.35 | 64.96 |
| RRF and rerank K=5 | 0.0174 | 0.0400 |
| RRF and rerank K=10 | 0.0217 | 0.0287 |
| RRF and rerank K=20 | 0.0246 | 0.0328 |
| MMR selection K=5 | 3.7984 | 4.6526 |
| MMR selection K=10 | 25.1158 | 40.0247 |
| MMR selection K=20 | 144.6209 | 147.4884 |
| MMR embedding (per query) | 5748.57 | 5903.18 |

## Recommended Candidate Depth

By unthresholded RRF NDCG, then MRR, then smaller K, use **vector K=10 and BM25 K=10** (NDCG 0.810, MRR 0.750). K=20 ties K=10 on this set but costs more. The strict >0.70 cutoff lowers Recall@K to 0.556 here; keep it only when precision is preferred over typo/synonym recall.

## Per-query RRF + Rerank

| K | Query | First relevant rank | Recall | NDCG |
|---:|---|---:|---:|---:|
| 5 | Titanium dioxide | 1 | 1.000 | 1.000 |
| 5 | titanium oxide | 1 | 1.000 | 1.000 |
| 5 | titanum dioxde | not found | 0.000 | 0.000 |
| 5 | wood alcohol | not found | 0.000 | 0.000 |
| 5 | methyl alcohol | not found | 0.000 | 0.000 |
| 5 | benzol | not found | 0.000 | 0.000 |
| 5 | Acetaldehyde | 1 | 1.000 | 1.000 |
| 5 | coal tar extract | 1 | 1.000 | 1.000 |
| 5 | formaldehyde gas | 1 | 1.000 | 1.000 |
| 10 | Titanium dioxide | 1 | 1.000 | 1.000 |
| 10 | titanium oxide | 1 | 1.000 | 1.000 |
| 10 | titanum dioxde | not found | 0.000 | 0.000 |
| 10 | wood alcohol | not found | 0.000 | 0.000 |
| 10 | methyl alcohol | not found | 0.000 | 0.000 |
| 10 | benzol | not found | 0.000 | 0.000 |
| 10 | Acetaldehyde | 1 | 1.000 | 1.000 |
| 10 | coal tar extract | 1 | 1.000 | 1.000 |
| 10 | formaldehyde gas | 1 | 1.000 | 1.000 |
| 20 | Titanium dioxide | 1 | 1.000 | 1.000 |
| 20 | titanium oxide | 1 | 1.000 | 1.000 |
| 20 | titanum dioxde | not found | 0.000 | 0.000 |
| 20 | wood alcohol | not found | 0.000 | 0.000 |
| 20 | methyl alcohol | not found | 0.000 | 0.000 |
| 20 | benzol | not found | 0.000 | 0.000 |
| 20 | Acetaldehyde | 1 | 1.000 | 1.000 |
| 20 | coal tar extract | 1 | 1.000 | 1.000 |
| 20 | formaldehyde gas | 1 | 1.000 | 1.000 |

## Local LLM-as-judge (K=20 RRF candidates)

The local Qwen judge labels query/candidate relevance as a diagnostic only; curated labels remain the benchmark ground truth.

- Agreement/accuracy vs curated labels: 0.633
- Precision: 0.120
- Recall: 1.000
- F1: 0.214
- Judge tokens: 4949 (4636 prompt, 313 completion)
- API cost: $0.00
