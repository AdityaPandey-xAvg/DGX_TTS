# TTS Benchmark — TTS_PHASE2

**Model:** `ai4bharat/IndicF5`  
**Reference audio:** `my_hindi.wav`  
**Timestamp:** run_151907

## Summary

- **Concurrency levels tested:** [1, 2, 4, 8]
- **Baseline concurrency 1 mean latency sec:** 2.7020
- **Baseline concurrency 1 p95 latency sec:** 4.2881
- **Best throughput concurrency level:** 1
- **Best throughput sentences per sec:** 0.3684
- **P95 latency degradation point concurrency:** 4
- **Quality mos stable across concurrency:** True
- **Quality cer stable across concurrency:** True
- **Recommended max concurrency:** 3

## Concurrency sweep results

| Concurrency | Sentences | Total (s) | Mean lat (s) | P95 lat (s) | Sent/s | RTF | UTMOS | IC-CER | W-CER | Peak VRAM | GPU% |
|-------------|-----------|-----------|--------------|-------------|--------|-----|-------|--------|-------|-----------|------|
| 1 | 8 | 21.71 | 2.702 | 4.288 | 0.37 | 0.652 | -1.00 | -1.000 | -1.000 | 2189 MB | 95.1 |
| 2 | 8 | 21.81 | 5.437 | 6.503 | 0.37 | 1.533 | -1.00 | -1.000 | -1.000 | 2283 MB | 96.9 |
| 4 | 8 | 21.95 | 10.955 | 11.986 | 0.36 | 3.218 | -1.00 | -1.000 | -1.000 | 2449 MB | 98.3 |
| 8 | 8 | 23.79 | 23.641 | 23.834 | 0.34 | 6.722 | -1.00 | -1.000 | -1.000 | 2759 MB | 90.3 |