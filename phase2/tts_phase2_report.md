# TTS Benchmark — TTS_PHASE2

**Model:** `ai4bharat/IndicF5`  
**Reference audio:** `my_hindi.wav`  
**Timestamp:** 2026-06-05T12:21:41.337454+00:00

## Summary

- **Concurrency levels tested:** [1, 2, 4, 8]
- **Baseline concurrency 1 mean latency sec:** 2.7702
- **Baseline concurrency 1 p95 latency sec:** 4.4158
- **Best throughput concurrency level:** 1
- **Best throughput sentences per sec:** 0.3605
- **P95 latency degradation point concurrency:** 4
- **Quality mos stable across concurrency:** True
- **Quality cer stable across concurrency:** True
- **Recommended max concurrency:** 3

## Concurrency sweep results

| Concurrency | Sentences | Total (s) | Mean lat (s) | P95 lat (s) | Sent/s | RTF | UTMOS | IC-CER | W-CER | Peak VRAM | GPU% |
|-------------|-----------|-----------|--------------|-------------|--------|-----|-------|--------|-------|-----------|------|
| 1 | 8 | 22.19 | 2.770 | 4.416 | 0.36 | 0.662 | -1.00 | -1.000 | -1.000 | 2189 MB | 94.3 |
| 2 | 8 | 22.26 | 5.557 | 6.630 | 0.36 | 1.491 | -1.00 | -1.000 | -1.000 | 2283 MB | 98.7 |
| 4 | 8 | 22.55 | 11.253 | 12.324 | 0.35 | 3.205 | -1.00 | -1.000 | -1.000 | 2451 MB | 96.8 |
| 8 | 8 | 22.76 | 22.684 | 22.784 | 0.35 | 6.481 | -1.00 | -1.000 | -1.000 | 2761 MB | 98.4 |