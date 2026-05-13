# Token Distribution — GRPO Summarization Eval Rollouts

> **Target:** 50 tokens per summary &nbsp;|&nbsp; **Within ±5** = 45–55 tokens &nbsp;|&nbsp; tokenizer: model-native

---

## Summary Table

| Model | Training Strategy | Reward Config | Mean Tokens | Median | Std | Within ±5 of 50 | Max-token hits |
|-------|-------------------|---------------|:-----------:|:------:|:---:|:---------------:|:--------------:|
| Qwen2.5-0.5B | Baseline | Base Model (no GRPO) | 78.7 | 58 | 71.5 | 32/200 (16%) | 3 |
| Qwen2.5-0.5B | Length Penalty Included | Length + BLEU | 67.2 | 64 | 16.7 | 36/200 (18%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length + BLEU+ROUGE | 71.0 | 66 | 26.6 | 33/200 (16%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length + METEOR | 67.2 | 65 | 18.2 | 40/200 (20%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length + METEOR+BLEU | 68.5 | 66 | 18.8 | 28/200 (14%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length + METEOR+ROUGE | 69.5 | 67 | 20.3 | 41/200 (20%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length + ROUGE-L | 62.9 | 61 | 16.9 | 45/200 (22%) | 0 |
| Qwen2.5-0.5B | Length Penalty Included | Length Only | 65.4 | 64 | 15.9 | 36/200 (18%) | 0 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality BLEU | 70.4 | 70 | 20.0 | 30/200 (15%) | 0 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality BLEU+ROUGE | 34.0 | 32 | 14.2 | 27/200 (14%) | 0 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality METEOR | 100.9 | 90 | 45.1 | 13/200 (6%) | 0 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality METEOR+BLEU | 123.8 | 112 | 61.7 | 8/200 (4%) | 1 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality METEOR+ROUGE | 65.8 | 62 | 28.8 | 44/200 (22%) | 0 |
| Qwen2.5-0.5B | Length Penalty Fine-tuned | Quality ROUGE | 31.9 | 29 | 11.6 | 20/200 (10%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + BLEU | 71.6 | 71 | 13.3 | 15/200 (8%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + BLEU+ROUGE | 68.8 | 67 | 12.9 | 26/200 (13%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + METEOR | 69.7 | 69 | 13.6 | 28/200 (14%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + METEOR+BLEU | 70.8 | 70 | 13.7 | 19/200 (10%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + METEOR+ROUGE | 71.5 | 72 | 14.7 | 29/200 (14%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length + ROUGE | 68.1 | 67 | 11.8 | 26/200 (13%) | 0 |
| LFM-2.5-350M | Length Penalty Included | Length Only | 70.1 | 69 | 11.8 | 16/200 (8%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality BLEU | 127.6 | 126 | 28.9 | 0/200 (0%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality BLEU+ROUGE | 32.4 | 29 | 13.0 | 20/200 (10%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality METEOR | 123.7 | 124 | 25.0 | 0/200 (0%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality METEOR+BLEU | 130.2 | 130 | 27.2 | 0/200 (0%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality METEOR+ROUGE | 71.9 | 72 | 20.0 | 33/200 (16%) | 0 |
| LFM-2.5-350M | Length Penalty Fine-tuned | Quality ROUGE | 27.9 | 27 | 8.9 | 5/200 (2%) | 0 |

---

## Qwen2.5-0.5B

### Qwen2.5-0.5B — Baseline — Base Model (no GRPO)

**n=200** &nbsp;|&nbsp; mean **78.7** &nbsp;|&nbsp; median **58** &nbsp;|&nbsp; std **71.5** &nbsp;|&nbsp; within ±5 of 50 tok: **32/200 (16%)** &nbsp;|&nbsp; hit 512-cap: **3**

```
     0–24       8  █████
    25–49      63  ████████████████████████████████████
    50–74      53  ██████████████████████████████ ◄
    75–99      33  ███████████████████
   100–124     21  ████████████
   125–149      8  █████
   150–174      3  ██
   175–199      2  █
   200–224      1  █
   225–249      2  █
   250–274      1  █
   275–299      2  █
   300–324      0  
   325–349      0  
   350–374      0  
   375–399      0  
   400–424      0  
   425–449      0  
   450–474      0  
   475–499      0  
   500–524      3  ██
```

### Qwen2.5-0.5B — Length Penalty Included — Length + BLEU

**n=200** &nbsp;|&nbsp; mean **67.2** &nbsp;|&nbsp; median **64** &nbsp;|&nbsp; std **16.7** &nbsp;|&nbsp; within ±5 of 50 tok: **36/200 (18%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      28  █████████
    50–74     108  ████████████████████████████████████ ◄
    75–99      55  ██████████████████
   100–124      8  ███
   125–149      1  
```

### Qwen2.5-0.5B — Length Penalty Included — Length + BLEU+ROUGE

**n=200** &nbsp;|&nbsp; mean **71.0** &nbsp;|&nbsp; median **66** &nbsp;|&nbsp; std **26.6** &nbsp;|&nbsp; within ±5 of 50 tok: **33/200 (16%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      30  ███████████
    50–74     101  ████████████████████████████████████ ◄
    75–99      45  ████████████████
   100–124     17  ██████
   125–149      4  █
   150–174      2  █
   175–199      0  
   200–224      0  
   225–249      0  
   250–274      1  
```

### Qwen2.5-0.5B — Length Penalty Included — Length + METEOR

**n=200** &nbsp;|&nbsp; mean **67.2** &nbsp;|&nbsp; median **65** &nbsp;|&nbsp; std **18.2** &nbsp;|&nbsp; within ±5 of 50 tok: **40/200 (20%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      27  █████████
    50–74     113  ████████████████████████████████████ ◄
    75–99      50  ████████████████
   100–124      7  ██
   125–149      2  █
   150–174      1  
```

### Qwen2.5-0.5B — Length Penalty Included — Length + METEOR+BLEU

**n=200** &nbsp;|&nbsp; mean **68.5** &nbsp;|&nbsp; median **66** &nbsp;|&nbsp; std **18.8** &nbsp;|&nbsp; within ±5 of 50 tok: **28/200 (14%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      29  ██████████
    50–74     106  ████████████████████████████████████ ◄
    75–99      51  █████████████████
   100–124     12  ████
   125–149      2  █
```

### Qwen2.5-0.5B — Length Penalty Included — Length + METEOR+ROUGE

**n=200** &nbsp;|&nbsp; mean **69.5** &nbsp;|&nbsp; median **67** &nbsp;|&nbsp; std **20.3** &nbsp;|&nbsp; within ±5 of 50 tok: **41/200 (20%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      28  ██████████
    50–74     105  ████████████████████████████████████ ◄
    75–99      51  █████████████████
   100–124     14  █████
   125–149      1  
   150–174      1  
```

### Qwen2.5-0.5B — Length Penalty Included — Length + ROUGE-L

**n=200** &nbsp;|&nbsp; mean **62.9** &nbsp;|&nbsp; median **61** &nbsp;|&nbsp; std **16.9** &nbsp;|&nbsp; within ±5 of 50 tok: **45/200 (22%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      45  ██████████████
    50–74     112  ████████████████████████████████████ ◄
    75–99      37  ████████████
   100–124      5  ██
   125–149      1  
```

### Qwen2.5-0.5B — Length Penalty Included — Length Only

**n=200** &nbsp;|&nbsp; mean **65.4** &nbsp;|&nbsp; median **64** &nbsp;|&nbsp; std **15.9** &nbsp;|&nbsp; within ±5 of 50 tok: **36/200 (18%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      30  █████████
    50–74     114  ████████████████████████████████████ ◄
    75–99      51  ████████████████
   100–124      5  ██
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality BLEU

**n=200** &nbsp;|&nbsp; mean **70.4** &nbsp;|&nbsp; median **70** &nbsp;|&nbsp; std **20.0** &nbsp;|&nbsp; within ±5 of 50 tok: **30/200 (15%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      31  █████████████
    50–74      88  ████████████████████████████████████ ◄
    75–99      69  ████████████████████████████
   100–124     10  ████
   125–149      2  █
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality BLEU+ROUGE

**n=200** &nbsp;|&nbsp; mean **34.0** &nbsp;|&nbsp; median **32** &nbsp;|&nbsp; std **14.2** &nbsp;|&nbsp; within ±5 of 50 tok: **27/200 (14%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24      51  ███████████████
    25–49     121  ████████████████████████████████████
    50–74      26  ████████ ◄
    75–99       2  █
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality METEOR

**n=200** &nbsp;|&nbsp; mean **100.9** &nbsp;|&nbsp; median **90** &nbsp;|&nbsp; std **45.1** &nbsp;|&nbsp; within ±5 of 50 tok: **13/200 (6%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       7  ████
    50–74      53  ███████████████████████████████ ◄
    75–99      61  ████████████████████████████████████
   100–124     31  ██████████████████
   125–149     25  ███████████████
   150–174      9  █████
   175–199      5  ███
   200–224      5  ███
   225–249      1  █
   250–274      1  █
   275–299      2  █
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality METEOR+BLEU

**n=200** &nbsp;|&nbsp; mean **123.8** &nbsp;|&nbsp; median **112** &nbsp;|&nbsp; std **61.7** &nbsp;|&nbsp; within ±5 of 50 tok: **8/200 (4%)** &nbsp;|&nbsp; hit 512-cap: **1**

```
     0–24       0  
    25–49       6  █████
    50–74      38  ██████████████████████████████ ◄
    75–99      45  ████████████████████████████████████
   100–124     26  █████████████████████
   125–149     31  █████████████████████████
   150–174     14  ███████████
   175–199     16  █████████████
   200–224     15  ████████████
   225–249      4  ███
   250–274      1  █
   275–299      3  ██
   300–324      0  
   325–349      0  
   350–374      0  
   375–399      0  
   400–424      0  
   425–449      0  
   450–474      0  
   475–499      0  
   500–524      1  █
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality METEOR+ROUGE

**n=200** &nbsp;|&nbsp; mean **65.8** &nbsp;|&nbsp; median **62** &nbsp;|&nbsp; std **28.8** &nbsp;|&nbsp; within ±5 of 50 tok: **44/200 (22%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       2  █
    25–49      55  ██████████████████████
    50–74      91  ████████████████████████████████████ ◄
    75–99      38  ███████████████
   100–124      8  ███
   125–149      3  █
   150–174      2  █
   175–199      0  
   200–224      0  
   225–249      0  
   250–274      0  
   275–299      1  
```

### Qwen2.5-0.5B — Length Penalty Fine-tuned — Quality ROUGE

**n=200** &nbsp;|&nbsp; mean **31.9** &nbsp;|&nbsp; median **29** &nbsp;|&nbsp; std **11.6** &nbsp;|&nbsp; within ±5 of 50 tok: **20/200 (10%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24      59  █████████████████
    25–49     123  ████████████████████████████████████
    50–74      18  █████ ◄
```

---

## LFM-2.5-350M

### LFM-2.5-350M — Length Penalty Included — Length + BLEU

**n=200** &nbsp;|&nbsp; mean **71.6** &nbsp;|&nbsp; median **71** &nbsp;|&nbsp; std **13.3** &nbsp;|&nbsp; within ±5 of 50 tok: **15/200 (8%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       7  ██
    50–74     118  ████████████████████████████████████ ◄
    75–99      68  █████████████████████
   100–124      7  ██
```

### LFM-2.5-350M — Length Penalty Included — Length + BLEU+ROUGE

**n=200** &nbsp;|&nbsp; mean **68.8** &nbsp;|&nbsp; median **67** &nbsp;|&nbsp; std **12.9** &nbsp;|&nbsp; within ±5 of 50 tok: **26/200 (13%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       7  ██
    50–74     134  ████████████████████████████████████ ◄
    75–99      54  ███████████████
   100–124      5  █
```

### LFM-2.5-350M — Length Penalty Included — Length + METEOR

**n=200** &nbsp;|&nbsp; mean **69.7** &nbsp;|&nbsp; median **69** &nbsp;|&nbsp; std **13.6** &nbsp;|&nbsp; within ±5 of 50 tok: **28/200 (14%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      16  █████
    50–74     115  ████████████████████████████████████ ◄
    75–99      63  ████████████████████
   100–124      6  ██
```

### LFM-2.5-350M — Length Penalty Included — Length + METEOR+BLEU

**n=200** &nbsp;|&nbsp; mean **70.8** &nbsp;|&nbsp; median **70** &nbsp;|&nbsp; std **13.7** &nbsp;|&nbsp; within ±5 of 50 tok: **19/200 (10%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      11  ████
    50–74     113  ████████████████████████████████████ ◄
    75–99      73  ███████████████████████
   100–124      3  █
```

### LFM-2.5-350M — Length Penalty Included — Length + METEOR+ROUGE

**n=200** &nbsp;|&nbsp; mean **71.5** &nbsp;|&nbsp; median **72** &nbsp;|&nbsp; std **14.7** &nbsp;|&nbsp; within ±5 of 50 tok: **29/200 (14%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       9  ███
    50–74     109  ████████████████████████████████████ ◄
    75–99      75  █████████████████████████
   100–124      7  ██
```

### LFM-2.5-350M — Length Penalty Included — Length + ROUGE

**n=200** &nbsp;|&nbsp; mean **68.1** &nbsp;|&nbsp; median **67** &nbsp;|&nbsp; std **11.8** &nbsp;|&nbsp; within ±5 of 50 tok: **26/200 (13%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       7  ██
    50–74     139  ████████████████████████████████████ ◄
    75–99      52  █████████████
   100–124      2  █
```

### LFM-2.5-350M — Length Penalty Included — Length Only

**n=200** &nbsp;|&nbsp; mean **70.1** &nbsp;|&nbsp; median **69** &nbsp;|&nbsp; std **11.8** &nbsp;|&nbsp; within ±5 of 50 tok: **16/200 (8%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       6  ██
    50–74     130  ████████████████████████████████████ ◄
    75–99      61  █████████████████
   100–124      3  █
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality BLEU

**n=200** &nbsp;|&nbsp; mean **127.6** &nbsp;|&nbsp; median **126** &nbsp;|&nbsp; std **28.9** &nbsp;|&nbsp; within ±5 of 50 tok: **0/200 (0%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       0  
    50–74       2  █ ◄
    75–99      28  ████████████████
   100–124     65  ████████████████████████████████████
   125–149     65  ████████████████████████████████████
   150–174     30  █████████████████
   175–199      7  ████
   200–224      2  █
   225–249      0  
   250–274      1  █
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality BLEU+ROUGE

**n=200** &nbsp;|&nbsp; mean **32.4** &nbsp;|&nbsp; median **29** &nbsp;|&nbsp; std **13.0** &nbsp;|&nbsp; within ±5 of 50 tok: **20/200 (10%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24      66  █████████████████████
    25–49     111  ████████████████████████████████████
    50–74      22  ███████ ◄
    75–99       1  
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality METEOR

**n=200** &nbsp;|&nbsp; mean **123.7** &nbsp;|&nbsp; median **124** &nbsp;|&nbsp; std **25.0** &nbsp;|&nbsp; within ±5 of 50 tok: **0/200 (0%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       0  
    50–74       2  █ ◄
    75–99      36  ██████████████████
   100–124     64  ████████████████████████████████
   125–149     71  ████████████████████████████████████
   150–174     23  ████████████
   175–199      3  ██
   200–224      1  █
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality METEOR+BLEU

**n=200** &nbsp;|&nbsp; mean **130.2** &nbsp;|&nbsp; median **130** &nbsp;|&nbsp; std **27.2** &nbsp;|&nbsp; within ±5 of 50 tok: **0/200 (0%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49       0  
    50–74       6  ███ ◄
    75–99      18  █████████
   100–124     60  ██████████████████████████████
   125–149     72  ████████████████████████████████████
   150–174     30  ███████████████
   175–199     14  ███████
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality METEOR+ROUGE

**n=200** &nbsp;|&nbsp; mean **71.9** &nbsp;|&nbsp; median **72** &nbsp;|&nbsp; std **20.0** &nbsp;|&nbsp; within ±5 of 50 tok: **33/200 (16%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24       0  
    25–49      24  █████████
    50–74      95  ████████████████████████████████████ ◄
    75–99      68  ██████████████████████████
   100–124      9  ███
   125–149      4  ██
```

### LFM-2.5-350M — Length Penalty Fine-tuned — Quality ROUGE

**n=200** &nbsp;|&nbsp; mean **27.9** &nbsp;|&nbsp; median **27** &nbsp;|&nbsp; std **8.9** &nbsp;|&nbsp; within ±5 of 50 tok: **5/200 (2%)** &nbsp;|&nbsp; hit 512-cap: **0**

```
     0–24      77  ███████████████████████
    25–49     120  ████████████████████████████████████
    50–74       2  █ ◄
    75–99       1  
```
