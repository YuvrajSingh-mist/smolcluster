# GRPO Evaluation Results — LFM 2.5 350M bf16

**Judge:** `gpt-5-mini-2025-08-07` · **Eval rounds:** 5 · **Examples:** 200 / 200 · **Dataset:** smoltldr test split  
**Metrics (G-Eval, each 0–1):** Faithfulness · Coverage · Conciseness · Clarity · **Composite (max 4.0)**

---

## Summary Table

| Reward Configuration | Composite | Faithfulness | Coverage | Conciseness | Clarity | Pass Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `length-quality-meteor-rouge` ⭐ | **2.701** | **0.834** | **0.493** | **0.685** | **0.690** | **45.2%** |
| `length-quality-bleu-rouge` | 2.387 | 0.696 | 0.443 | 0.606 | 0.643 | 35.4% |
| `length-quality-meteor-bleu` | 2.377 | 0.696 | 0.451 | 0.595 | 0.634 | 34.2% |
| `length-quality-meteor` | 2.358 | 0.689 | 0.433 | 0.595 | 0.641 | 32.5% |
| `length-quality-rouge` | 2.278 | 0.642 | 0.414 | 0.575 | 0.646 | 30.1% |
| `length-quality-bleu` | 2.243 | 0.620 | 0.401 | 0.556 | 0.665 | 26.7% |
| `length-only` (baseline) | 2.233 | 0.627 | 0.378 | 0.554 | 0.674 | 24.6% |

---

## Delta vs `length-only` Baseline

**Test:** Paired t-test, n = 200, α = 0.05 · **p-values:** one-sided (candidate > baseline)

| Reward Configuration | ΔComposite | p (composite) | Sig | ΔFaithfulness | Sig | ΔCoverage | Sig | ΔConciseness | Sig | ΔClarity | Sig | Sig Metrics |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `length-quality-meteor-rouge` ⭐ | **+0.469** | **2.82e-21** | ✅ | **+0.207** | ✅ | **+0.115** | ✅ | **+0.131** | ✅ | +0.015 | ❌ | **4/5** |
| `length-quality-bleu-rouge` | +0.154 | 6.07e-04 | ✅ | +0.069 | ✅ | +0.065 | ✅ | +0.052 | ✅ | -0.031 | ❌ | 4/5 |
| `length-quality-meteor-bleu` | +0.144 | 2.47e-03 | ✅ | +0.069 | ✅ | +0.073 | ✅ | +0.041 | ✅ | -0.04ı0 | ❌ | 4/5 |
| `length-quality-meteor` | +0.125 | 6.12e-03 | ✅ | +0.062 | ✅ | +0.056 | ✅ | +0.041 | ✅ | -0.033 | ❌ | 4/5 |
| `length-quality-rouge` | +0.045 | 0.162 | ❌ | +0.016 | ❌ | +0.036 | ✅ | +0.021 | ❌ | -0.028 | ❌ | 1/5 |
| `length-quality-bleu` | +0.010 | 0.414 | ❌ | -0.006 | ❌ | +0.024 | ✅ | +0.002 | ❌ | -0.009 | ❌ | 1/5 |

---

## Per-Configuration Details

### 1. `length-quality-meteor-rouge` ⭐ Best

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.834 | 94.9% |
| Coverage | 0.493 | 50.7% |
| Conciseness | 0.685 | 81.8% |
| Clarity | 0.690 | 85.6% |
| **Composite** | **2.701** | — |
| **Overall Pass Rate** | — | **45.2%** |

> Dominant across all metrics. Faithfulness jumps ~+21 points over baseline and ~+14 points over any other multi-metric config. The combination of METEOR (semantic recall) + ROUGE (n-gram precision) appears to create a well-calibrated quality signal for this architecture.

---

### 2. `length-quality-bleu-rouge`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.696 | 76.9% |
| Coverage | 0.443 | 41.2% |
| Conciseness | 0.606 | 68.9% |
| Clarity | 0.643 | 79.7% |
| **Composite** | **2.387** | — |
| **Overall Pass Rate** | — | **35.4%** |

---

### 3. `length-quality-meteor-bleu`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.696 | 77.0% |
| Coverage | 0.451 | 41.1% |
| Conciseness | 0.595 | 67.5% |
| Clarity | 0.634 | 79.4% |
| **Composite** | **2.377** | — |
| **Overall Pass Rate** | — | **34.2%** |

---

### 4. `length-quality-meteor`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.689 | 77.2% |
| Coverage | 0.433 | 37.1% |
| Conciseness | 0.595 | 65.7% |
| Clarity | 0.641 | 77.4% |
| **Composite** | **2.358** | — |
| **Overall Pass Rate** | — | **32.5%** |

---

### 5. `length-quality-rouge`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.642 | 69.8% |
| Coverage | 0.414 | 34.4% |
| Conciseness | 0.575 | 64.4% |
| Clarity | 0.646 | 81.8% |
| **Composite** | **2.278** | — |
| **Overall Pass Rate** | — | **30.1%** |

---

### 6. `length-quality-bleu`

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.620 | 69.5% |
| Coverage | 0.401 | 31.3% |
| Conciseness | 0.556 | 61.0% |
| Clarity | 0.665 | 82.9% |
| **Composite** | **2.243** | — |
| **Overall Pass Rate** | — | **26.7%** |

---

### 7. `length-only` (Baseline)

| Metric | Mean Score | Pass Rate |
|---|:---:|:---:|
| Faithfulness | 0.627 | 70.2% |
| Coverage | 0.378 | 27.1% |
| Conciseness | 0.554 | 61.4% |
| Clarity | 0.674 | 87.4% |
| **Composite** | **2.233** | — |
| **Overall Pass Rate** | — | **24.6%** |

---

## Evaluation Cost & Runtime

| Configuration | Cost (USD) | Judge Time (h) |
|---|:---:|:---:|
| `length-only` | $6.96 | 2.58 h |
| `length-quality-bleu` | $7.28 | 2.63 h |
| `length-quality-rouge` | $7.38 | 3.38 h |
| `length-quality-meteor` | $7.28 | 2.58 h |
| `length-quality-bleu-rouge` | $7.26 | 2.50 h |
| `length-quality-meteor-bleu` | $7.32 | 2.67 h |
| `length-quality-meteor-rouge` | $7.14 | 2.77 h |
| **Total** | **$50.60** | **~18.9 h** |
