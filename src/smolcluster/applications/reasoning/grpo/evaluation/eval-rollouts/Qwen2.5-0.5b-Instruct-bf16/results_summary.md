# GRPO Evaluation Results — Qwen 2.5 0.5B Instruct bf16

**Judge:** `gpt-5-mini-2025-08-07` · **Eval rounds:** 5 · **Examples:** 200 / 200 · **Dataset:** smoltldr test split  
**Metrics (G-Eval, each 0–1):** Faithfulness · Coverage · Conciseness · Clarity · **Composite (max 4.0)**

---

## Summary Table

| Reward Configuration | Composite | Faithfulness | Coverage | Conciseness | Clarity | Pass Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `length-quality-meteor-rouge` ⭐ | **2.769** | **0.832** | **0.511** | **0.659** | **0.767** | **44.3%** |
| `length-quality-bleu-rouge` | 2.732 | 0.810 | 0.502 | 0.650 | 0.770 | 39.1% |
| `length-quality-meteor-bleu` | 2.664 | 0.792 | 0.468 | 0.648 | 0.756 | 38.3% |
| `length-quality-rouge-l` | 2.555 | 0.725 | 0.415 | 0.637 | 0.778 | 32.4% |
| `length-quality-meteor` | 2.484 | 0.721 | 0.427 | 0.625 | 0.711 | — |
| `length-quality-bleu` | 2.400 | 0.680 | 0.399 | 0.577 | 0.744 | 26.9% |
| `length-only` (baseline) | 2.416 | 0.678 | 0.407 | 0.592 | 0.739 | 30.7% |

---

## Delta vs `length-only` Baseline

**Test:** Paired t-test, n = 200, α = 0.05 · **p-values:** one-sided (candidate > baseline)

| Reward Configuration | ΔComposite | p (composite) | Sig | ΔFaithfulness | Sig | ΔCoverage | Sig | ΔConciseness | Sig | ΔClarity | Sig | Sig Metrics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `length-quality-meteor-rouge` ⭐ | **+0.353** | **6.73e-10** | ✅ | **+0.154** | ✅ | **+0.103** | ✅ | **+0.067** | ✅ | +0.029 | ✅ | **5/5** |
| `length-quality-bleu-rouge` | +0.316 | 1.39e-07 | ✅ | +0.132 | ✅ | +0.095 | ✅ | +0.058 | ✅ | +0.032 | ✅ | 5/5 |
| `length-quality-meteor-bleu` | +0.249 | 1.08e-05 | ✅ | +0.114 | ✅ | +0.061 | ✅ | +0.056 | ✅ | +0.018 | ❌ | 4/5 |
| `length-quality-rouge-l` | +0.139 | 4.23e-03 | ✅ | +0.047 | ✅ | +0.008 | ❌ | +0.045 | ✅ | +0.039 | ✅ | 4/5 |
| `length-quality-meteor` | +0.069 | 0.106 | ❌ | +0.043 | ✅ | +0.020 | ❌ | +0.033 | ✅ | -0.028 | ❌ | 2/5 |
| `length-quality-bleu` | -0.015 | 0.615 | ❌ | +0.002 | ❌ | -0.008 | ❌ | -0.016 | ❌ | +0.006 | ❌ | 0/5 |

---

## Per-Configuration Details

### 1. `length-quality-meteor-rouge` ⭐ Best

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.832 | 92.8% |
| Coverage | 0.511 | 52.9% |
| Conciseness | 0.659 | 78.2% |
| Clarity | 0.767 | 92.3% |
| **Composite** | **2.769** | — |
| **Overall Pass Rate** | — | **44.3%** |

> Best composite and only config to achieve significance on all 5 metrics. Largest individual gains on Faithfulness (+0.154) and Coverage (+0.103).

---

### 2. `length-quality-bleu-rouge`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.810 | 89.7% |
| Coverage | 0.502 | 51.0% |
| Conciseness | 0.650 | 76.1% |
| Clarity | 0.770 | 92.6% |
| **Composite** | **2.732** | — |
| **Overall Pass Rate** | — | **39.1%** |

> Very close to meteor-rouge (Δ0.037 composite). Also 5/5 significant. Slightly higher Clarity.

---

### 3. `length-quality-meteor-bleu`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.792 | 89.6% |
| Coverage | 0.468 | 45.7% |
| Conciseness | 0.648 | 77.5% |
| Clarity | 0.756 | 92.6% |
| **Composite** | **2.664** | — |
| **Overall Pass Rate** | — | **38.3%** |

> 4/5 significant; Clarity falls just short (p = 0.107).

---

### 4. `length-quality-rouge-l`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.725 | 79.1% |
| Coverage | 0.415 | 36.0% |
| Conciseness | 0.637 | 73.0% |
| Clarity | 0.778 | 92.8% |
| **Composite** | **2.555** | — |
| **Overall Pass Rate** | — | **32.4%** |

> 4/5 significant; Coverage not significant (p = 0.293). Strong Clarity gains.

---

### 5. `length-quality-meteor`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.721 | — |
| Coverage | 0.427 | — |
| Conciseness | 0.625 | — |
| Clarity | 0.711 | — |
| **Composite** | **2.484** | — |
| **Overall Pass Rate** | — | — |

> Composite improvement not significant (p = 0.106). Only Faithfulness and Conciseness are individually significant; Clarity regresses.

---

### 6. `length-quality-bleu`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.680 | 76.3% |
| Coverage | 0.399 | 31.6% |
| Conciseness | 0.577 | 64.2% |
| Clarity | 0.744 | 90.1% |
| **Composite** | **2.400** | — |
| **Overall Pass Rate** | — | **26.9%** |

> **0/5 significant.** Composite is actually slightly below baseline (−0.015). BLEU-only reward provides no meaningful quality improvement for this model.

---

### 7. `length-only` (Baseline)

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.678 | 73.1% |
| Coverage | 0.407 | 33.9% |
| Conciseness | 0.592 | 68.1% |
| Clarity | 0.739 | 89.4% |
| **Composite** | **2.416** | — |
| **Overall Pass Rate** | — | **30.7%** |

---



## Evaluation Cost & Runtime

| Configuration | Cost (USD) | Judge Time (h) |
|---|:---:|:---:|
| `length-only` | $7.00 | 14.7 h |
| `length-quality-bleu` | $6.91 | 2.9 h |
| `length-quality-rouge-l` | $6.88 | 15.4 h |
| `length-quality-meteor` | — | — |
| `length-quality-bleu-rouge` | $6.93 | 2.0 h |
| `length-quality-meteor-bleu` | $7.05 | 2.8 h |
| `length-quality-meteor-rouge` | $6.95 | 2.0 h |
