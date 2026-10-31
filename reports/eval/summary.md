# Evaluation report

Golden set: 55 questions (49 in scope, 6 out of scope).

## Retrieval

Top 10 passages per question, collapsed to articles (any slice of a long article counts as the article); expected articles have grade 2 and acceptable ones grade 1. Configurations search the raw question, without the agent's query analysis, except `dense+analyzer` and `hybrid+analyzer`, which search the analyzer's queries. `hybrid+rerank` reranks the first 20 passages of its search with the cross-encoder and keeps the best 10.

| Config | n | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense` | 49 | 0.71 [0.59, 0.84] | 0.96 [0.90, 1.00] | 0.82 [0.72, 0.89] | 0.84 [0.75, 0.91] | 0.82 [0.73, 0.90] | 0.71 [0.63, 0.77] | 60 |
| `sparse` | 49 | 0.27 [0.14, 0.39] | 0.55 [0.41, 0.69] | 0.43 [0.30, 0.54] | 0.56 [0.43, 0.68] | 0.38 [0.27, 0.50] | 0.38 [0.29, 0.47] | 65 |
| `hybrid` | 49 | 0.57 [0.43, 0.71] | 0.96 [0.90, 1.00] | 0.80 [0.71, 0.87] | 0.85 [0.77, 0.92] | 0.72 [0.62, 0.81] | 0.64 [0.57, 0.71] | 146 |
| `hybrid+rerank` | 49 | 0.65 [0.51, 0.78] | 0.92 [0.84, 0.98] | 0.79 [0.69, 0.87] | 0.83 [0.74, 0.91] | 0.77 [0.66, 0.86] | 0.68 [0.61, 0.76] | 12072 |
| `dense+analyzer` | 49 | 0.69 [0.57, 0.82] | 0.96 [0.90, 1.00] | 0.86 [0.77, 0.93] | 0.86 [0.78, 0.94] | 0.82 [0.73, 0.90] | 0.75 [0.68, 0.81] | 47 |
| `hybrid+analyzer` | 49 | 0.65 [0.51, 0.78] | 0.98 [0.94, 1.00] | 0.88 [0.81, 0.95] | 0.91 [0.84, 0.97] | 0.80 [0.71, 0.88] | 0.76 [0.70, 0.82] | 173 |

Paired differences in nDCG@10 against `dense`:

| Config | Mean diff | 95% CI | P(diff <= 0) |
|---|---:|---:|---:|
| `sparse` | -0.326 | [-0.407, -0.240] | 1.000 |
| `hybrid` | -0.065 | [-0.103, -0.029] | 1.000 |
| `hybrid+rerank` | -0.023 | [-0.062, +0.014] | 0.883 |
| `dense+analyzer` | +0.044 | [-0.010, +0.099] | 0.059 |
| `hybrid+analyzer` | +0.055 | [-0.016, +0.137] | 0.065 |

nDCG@10 by question type:

| Type | n | `dense` | `sparse` | `hybrid` | `hybrid+rerank` | `dense+analyzer` | `hybrid+analyzer` |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 4 | 0.97 | 0.82 | 0.92 | 0.95 | 0.98 | 0.96 |
| conceptual | 16 | 0.65 | 0.37 | 0.60 | 0.63 | 0.66 | 0.72 |
| cross_law | 8 | 0.58 | 0.34 | 0.54 | 0.63 | 0.64 | 0.69 |
| lookup | 8 | 0.79 | 0.38 | 0.70 | 0.76 | 0.82 | 0.79 |
| multi_article | 10 | 0.68 | 0.16 | 0.55 | 0.61 | 0.80 | 0.77 |
| temporal | 3 | 0.88 | 0.73 | 0.90 | 0.83 | 0.88 | 0.80 |

Unit recall@10 (questions that need a specific inciso or paragraph of a long article: share of those units among the retrieved slices):

| Config | n | Unit recall@10 |
|---|---:|---:|
| `dense` | 8 | 0.38 [0.12, 0.75] |
| `sparse` | 8 | 0.25 [0.00, 0.62] |
| `hybrid` | 8 | 0.31 [0.06, 0.62] |
| `hybrid+rerank` | 8 | 0.31 [0.06, 0.62] |
| `dense+analyzer` | 8 | 0.69 [0.38, 1.00] |
| `hybrid+analyzer` | 8 | 0.75 [0.38, 1.00] |

_Generated 2026-10-31T14:18:12+00:00. Values in brackets are 95% bootstrap confidence intervals over questions._
