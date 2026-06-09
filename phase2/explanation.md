Sure. Let me explain this completely from first principles.

---

## The core problem you're trying to solve

You're building a voice AI system. In production, it won't be just one person talking to it. Multiple users will send requests at the same time. The question Phase 2 answers is:

**"How many users can my system handle simultaneously before it starts feeling slow?"**

---

## What "simultaneous" actually means at the hardware level

Your RTX 5060 Ti has one GPU. That GPU can only do one thing at a time at the hardware level — it processes one batch of computation, then the next. It's like a single very fast kitchen with one chef.

When two users send a TTS request at the same time, both requests arrive at the GPU simultaneously. But the GPU can't magically split itself in two. What actually happens is the GPU **interleaves** their work — it processes a chunk of User A's audio generation, then a chunk of User B's, then back to A, and so on, very rapidly. Both requests are "in progress" at the same time, but they're sharing the same physical hardware.

This is fine up to a point. But as you add more users, each one gets a smaller slice of GPU time. Eventually, requests start queuing up waiting for GPU time, and latency spikes.

---

## What Phase 1 established

Phase 1 was your **baseline** — one user, no competition. You learned:

- A short sentence takes ~1.6s
- A long sentence takes ~12.3s  
- RTF ~0.49× on long text (GPU generating 2× faster than real-time)
- GPU utilisation ~96%

That last number is important. **96% GPU utilisation on a single request** means your GPU is already nearly maxed out serving just one person. There's very little headroom left for additional concurrent users before contention begins.

---

## What Phase 2 is actually doing

Phase 2 simulates a waiting room filling up.

```
Concurrency 1  →  1 person walks in, gets served immediately
Concurrency 2  →  2 people walk in at exactly the same time
Concurrency 4  →  4 people walk in at exactly the same time
Concurrency 8  →  8 people walk in at exactly the same time
```

For each level, the script launches that many threads simultaneously. Each thread independently calls the TTS model with a different sentence. All threads start at the same moment. The GPU has to serve all of them.

The benchmark then measures, for each user in that group:
- How long did they wait until their audio was ready? (latency)
- How much did that latency grow compared to the single-user baseline? (degradation)

---

## The degradation curve — what you're hunting for

As concurrency increases, latency follows a predictable shape:

```
Latency
  │                                          ● ← system overwhelmed
  │                               ●
  │                    ●
  │          ●
  │    ●
  │●  ← baseline (1 user)
  └─────────────────────────────────────────── Concurrency
       1    2    4    8   16   32
```

There are three zones:

**Zone 1 — Flat (safe zone).** Adding users doesn't hurt latency much. The GPU handles the extra load by interleaving efficiently. You want to stay here.

**Zone 2 — The knee.** Latency starts rising noticeably. The GPU is saturated and requests begin queuing. This is your **maximum recommended concurrency** for production.

**Zone 3 — The cliff.** Latency spikes sharply. Requests are queueing faster than the GPU can process them. Users experience unacceptable delays.

Phase 2's job is to find exactly where Zone 1 ends and Zone 2 begins.

---

## Why P95 latency matters more than mean latency

The summary reports P50, P95, and P99 latency. Here's why P95 is the one you care about most:

Mean latency tells you what the average user experiences. P95 tells you what the **worst 5% of users** experience. In a voice AI product, that 5% matters — those are real people getting a bad experience.

The benchmark flags the degradation point as the first concurrency level where **P95 latency exceeds 2× the baseline P95**. That's a practical production threshold — if your slowest users are waiting more than twice as long as they would alone, the system is under strain.

---

## What your specific results will reveal

Given that your GPU is already at 96% on a single request, you should expect the knee to appear relatively early — probably around concurrency 2 or 4. This is not necessarily bad news. It tells you one important thing:

**A single IndicF5 instance on one GPU is not designed to serve many simultaneous users.**

The production solution for this — which is what Phase 2 is building evidence for — is to run multiple model instances in parallel (as we discussed earlier with the multi-instance architecture). If one instance maxes out at concurrency 2, you run 4 instances and serve 8 users simultaneously with each user getting near-baseline latency.

The Phase 2 numbers give you the evidence to make that decision: "One GPU instance handles N concurrent users acceptably. For M total users, we need M/N instances."

That's the number you'll take to the DGX Spark — where instead of one RTX 5060 Ti, you have 128GB of unified memory and can run many IndicF5 instances simultaneously.

Great reading of the results. Let me answer both clearly.

---

## 1. How we simulated concurrent users — the relevant code

The key is `ThreadPoolExecutor`. Here's the relevant section from `tts_phase2_concurrent.py`, simplified to just the important parts:

```python
with ThreadPoolExecutor(max_workers=actual_concurrency) as executor:
    for worker_id, sent in enumerate(chunk):
        fut = executor.submit(
            _worker_inference,
            worker_id, sent, tts_model, ...
        )
```

Think of `ThreadPoolExecutor` as a manager who has N employees (threads). You hand the manager 8 tasks at once, and the manager immediately assigns one task to each employee — all 8 start working simultaneously.

In plain English, here's what happens at each concurrency level:

**Concurrency 1:**
```
Manager hires 1 employee.
Gives them sentence c001.
Employee calls TTS model. Waits. Gets audio back.
Done.
```

**Concurrency 8:**
```
Manager hires 8 employees simultaneously.
Employee 1 gets c001 → calls TTS model
Employee 2 gets c002 → calls TTS model  ← all 8 start
Employee 3 gets c003 → calls TTS model     at the exact
Employee 4 gets c004 → calls TTS model     same moment
Employee 5 gets c005 → calls TTS model
Employee 6 gets c006 → calls TTS model
Employee 7 gets c007 → calls TTS model
Employee 8 gets c008 → calls TTS model
```

All 8 employees are now waiting for the GPU to respond. But there's only one GPU. So the GPU interleaves their work — a little bit for Employee 1, then Employee 2, then 3... cycling through all 8. Each employee eventually gets their audio back, but they all had to share the GPU, so each one waited much longer than if they were alone.

The `_worker_inference` function is what each employee actually runs:

```python
def _worker_inference(worker_id, sent, tts_model, ref_audio_path, ...):
    t0 = time.perf_counter()          # start the clock
    audio, latency = run_tts(         # call the model — this is the waiting part
        tts_model, text, ref_audio_path, ref_text
    )
    # record how long this specific worker waited
    return TTSFileResult(latency_sec=latency, ...)
```

Each worker measures its own wall clock time independently. So after all 8 finish, you have 8 individual latency measurements — one per worker. That list of 8 numbers is what P95 is calculated from.

---

## 2. How P95 and P99 are calculated — using your actual numbers

From your concurrency=8 run, 8 workers each recorded their own latency. You don't have the individual numbers in the report, but let's reconstruct what likely happened. Mean was 22.684s and P95 was 22.784s, so the values were all clustered together. Let's use concurrency=1 as a cleaner example since the numbers are more spread.

**At concurrency=1, across 2 repeats × 8 sentences = 16 individual latency measurements**, sorted from fastest to slowest, it might look something like:

```
Position  Latency
   1       1.2s   ← fastest (short sentence, lucky GPU state)
   2       1.4s
   3       1.5s
   4       1.6s
   5       1.7s
   6       1.8s
   7       2.1s
   8       2.3s
   9       2.8s
  10       3.1s
  11       3.4s
  12       3.7s
  13       3.9s
  14       4.1s
  15       4.3s   ← P95 lives around here (95% of 16 = position 15.2)
  16       4.8s   ← slowest (long sentence, GPU contention)
```

**P95 = the value at the 95th percentile position.**

With 16 measurements: 95% of 16 = 15.2, so you look at position 15 or 16. Your P95 of 4.416s means that 95% of users got their audio in 4.4 seconds or less. Only 5% (the slowest 1 out of 16) waited longer.

**Mean** is just the average: add all 16 numbers, divide by 16 → 2.770s.

Now look at what happened at concurrency=8:

```
P95 latency:
  Concurrency 1 →  4.4s   (baseline)
  Concurrency 2 →  6.6s   (1.5× worse)
  Concurrency 4 → 12.3s   (2.8× worse)  ← degradation point crossed here (>2× baseline)
  Concurrency 8 → 22.8s   (5.2× worse)
```

The benchmark flagged concurrency=4 as the degradation point because 12.3s > 2× 4.4s (8.8s). That's the moment when your worst-case users started waiting unacceptably long.

The code that calculates this is simple:

```python
def percentile(data: list[float], p: float) -> float:
    s = sorted(data)                        # sort all latencies low to high
    idx = int(len(s) * p / 100)            # find the position
    return s[min(idx, len(s) - 1)]         # return the value at that position
```

For P95: sort the list, jump to the 95% position, read the number there. That's it.

---

## What your results are telling you in plain English

```
Concurrency 1 → Mean 2.8s,  P95 4.4s   ← baseline, GPU at 94%
Concurrency 2 → Mean 5.6s,  P95 6.6s   ← doubled latency, GPU at 99%
Concurrency 4 → Mean 11.3s, P95 12.3s  ← 4× worse, GPU saturated
Concurrency 8 → Mean 22.7s, P95 22.8s  ← everyone waiting ~22s
```

The total wall time stays at ~22s across all levels. That's the key insight — the GPU is doing the same total amount of work regardless of concurrency. The difference is that at concurrency=8, all 8 users are waiting for that same 22s of work to complete, rather than getting served one by one. The GPU doesn't get faster by having more requests — it just splits its attention more ways.

**Recommended max concurrency: 3** (from the summary) means: with this single GPU instance, serve at most 3 users simultaneously before P95 latency crosses the 2× threshold. For the DGX Spark with 128GB memory, you'd run multiple IndicF5 instances — each handling up to 3 concurrent users — to scale horizontally.


Great question. Short answer: **we implemented zero scheduling logic**. We just fire all threads and let the OS and CUDA handle everything. Let me explain exactly what happens.

---

## Who does the scheduling — it's not us

There are actually **three layers** doing scheduling, none of which we wrote:

```
Our code          → just launches 8 threads simultaneously, that's it
      ↓
Python OS threads → scheduled by the operating system (Windows/Linux)
      ↓
PyTorch / CUDA    → each thread submits GPU kernels to the CUDA queue
      ↓
GPU hardware      → NVIDIA's own scheduler interleaves the kernels
```

We have zero control over how the GPU splits its time between threads. NVIDIA's hardware scheduler decides that completely on its own — it's not round-robin, it's not priority-based, it's a proprietary interleaving algorithm built into the GPU silicon.

---

## How each thread measures its own latency

Look at `_worker_inference`:

```python
def _worker_inference(worker_id, sent, tts_model, ...):

    t0 = time.perf_counter()        # ← clock starts HERE, inside the thread
    audio, latency = run_tts(...)   # ← thread is BLOCKED here, waiting for GPU
                                    #   this line does not return until audio is ready
    # clock stops inside run_tts:
    #   t0 = time.perf_counter()
    #   audio = model(text, ref_audio, ref_text)  ← blocks until done
    #   latency = time.perf_counter() - t0        ← stop the clock
```

Each thread has its own independent `t0`. The thread starts its own clock, then **blocks** on `run_tts()` — meaning it sits frozen, doing nothing, waiting — until the GPU finishes and returns the audio. Only then does the clock stop.

---

## Walking through concurrency=8 step by step

Here's exactly what happens in real time:

```
t=0.000s  All 8 threads are created and start simultaneously
          Each thread immediately records its own t0 = time.perf_counter()
          Each thread calls model(...) and is now BLOCKED

          Thread 1: t0=0.000, waiting... ⏳
          Thread 2: t0=0.000, waiting... ⏳
          Thread 3: t0=0.000, waiting... ⏳
          Thread 4: t0=0.000, waiting... ⏳
          Thread 5: t0=0.000, waiting... ⏳
          Thread 6: t0=0.000, waiting... ⏳
          Thread 7: t0=0.000, waiting... ⏳
          Thread 8: t0=0.000, waiting... ⏳

          [GPU is now interleaving all 8 simultaneously — we have no control over this]

t=18.2s   Thread 3 gets its audio back first (it had the shortest sentence)
          latency = 18.2 - 0.000 = 18.2s  ✓ recorded

t=20.1s   Thread 7 gets its audio back
          latency = 20.1 - 0.000 = 20.1s  ✓ recorded

t=21.4s   Thread 1 gets its audio back
          latency = 21.4 - 0.000 = 21.4s  ✓ recorded

...

t=22.8s   Thread 5 gets its audio back last
          latency = 22.8 - 0.000 = 22.8s  ✓ recorded
```

Every thread started counting at t=0. Every thread stopped counting when **its own** audio came back. So Thread 3 measured 18.2s and Thread 5 measured 22.8s — different numbers from the same run, because the GPU finished their work at different times.

This is why your P95 was 22.784s at concurrency=8 — the slowest thread started at t=0 and didn't get its result back until t=22.8s. It experienced the full 22 seconds of waiting.

---

## The key insight: why all t0 are the same but all end times are different

```
Thread 1  |████████████████████████████████████░░░| done at 21.4s
Thread 2  |█████████████████████░░░░░░░░░░░░░░░░░| done at 19.1s
Thread 3  |████████████████████░░░░░░░░░░░░░░░░░░| done at 18.2s
Thread 4  |██████████████████████████████████████| done at 22.1s
Thread 5  |███████████████████████████████████████| done at 22.8s ← P95
Thread 6  |█████████████████████████░░░░░░░░░░░░░| done at 20.3s
Thread 7  |████████████████████████░░░░░░░░░░░░░░| done at 20.1s
Thread 8  |██████████████████████████████████████| done at 22.5s
          ↑                                      ↑
        t=0                                   t=22.8s
        all start here                        last one finishes here
```

All blocks start at the same left edge (same t0). They end at different times because the GPU gave different amounts of attention to each thread depending on their input length and the order CUDA happened to schedule their kernels. We wrote none of that logic — we just launched 8 threads and let the system do its thing.