#!/usr/bin/env python3
"""
tts_phase2_concurrent.py — TTS Concurrency Stress Benchmarking
===============================================================

WHY THIS DIFFERS FROM STT PHASE 2 (important to understand)
─────────────────────────────────────────────────────────────
STT Phase 2 used TRUE GPU batching: one forward pass, B inputs, processed
in parallel inside a single CUDA kernel. This works because the inputs
(audio tensors) can be padded to the same shape.

IndicF5 (F5-TTS based) does NOT expose a public batch API. Its inference
pipeline internally: tokenises text → computes mel frames → runs flow-
matching ODE solver → decodes to waveform. Each call is a complete
independent pipeline run. You cannot simply stack N text inputs into one
tensor without reimplementing the internals.

WHAT WE DO INSTEAD: Concurrent worker simulation
  We launch C concurrent threads, each running one IndicF5 inference call.
  This simulates C simultaneous users hitting the same model instance.

  On the DGX Spark (unified 128 GB memory), the GPU scheduler interleaves
  CUDA kernels from different threads. This is NOT the same as batching,
  but it IS realistic — it's how a production TTS service actually handles
  concurrent requests when running a single model instance.

WHAT THIS MEASURES:
  - How latency degrades as concurrency increases (P50 vs P95 vs P99)
  - The throughput ceiling (sentences/sec) under concurrent load
  - GPU utilisation under concurrent pressure
  - Quality stability — MOS, CER, WER should not degrade under load
    (if they do, the model is numerically unstable under threading)
  - The concurrency level where P95 latency crosses an acceptable threshold

CONCURRENCY LEVELS SWEPT: [1, 2, 4, 8, 16, 32]
  Level 1 = baseline (matches Phase 1 sequential)
  Level 32 = stress test (32 simultaneous users)

Usage:
    python tts_phase2_concurrent.py \
        --ref-audio   ./ref_clips/ref_speaker.wav \
        --ref-text    "transcript of ref audio" \
        --sentences   ./test_sentences.json \
        --output      ./tts_results/phase2 \
        --concurrency 1 2 4 8 16 32 \
        --repeats     3

    # Smoke test:
    python tts_phase2_concurrent.py \
        --ref-audio   ./ref_clips/ref_speaker.wav \
        --ref-text    " हलो मेरा नाम आदित्य है।" \
        --concurrency 1 4 8 \
        --repeats     1 \
        --limit       8 \
        --skip-roundtrip
    # Smoke test:
    python tts_phase2_concurrent.py \
        --ref-audio   ./ref_clips/ref_speaker.wav \
        --ref-text    " हलो मेरा नाम आदित्य है।" \
        --concurrency 1 4 8 \
        --repeats     1 \
        --limit       8 \
        --skip-roundtrip
   python tts_phase2_concurrent.py \
    --ref-audio   ./my_hindi.wav \
    --ref-text     " हलो मेरा नाम आदित्य है।"\
    --sentences   ./long_test.json \
    --output      ./tts_results/phase2 \
    --concurrency 1 2 4 8 \
    --repeats     2 \
    --skip-roundtrip 


Output:
    tts_phase2_results.json   — per-concurrency aggregated results
    tts_phase2_report.md      — comparison table
    audio/                    — all generated WAVs (sent_id_c{N}_w{W}.wav)
"""

import argparse
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tts_benchmark_base import (
    GPUMonitor,
    TTSBatchResult,
    TTSFileResult,
    TTSPhaseReport,
    audio_duration,
    clipping_ratio,
    estimate_mos,
    load_indicconformer_stt,
    load_indicf5,
    load_test_sentences,
    load_utmos,
    load_whisper_stt,
    round_trip_stt,
    run_tts,
    save_audio,
    save_tts_report,
    silence_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker inference function (runs in a thread)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_inference(
    worker_id: int,
    sent: dict,
    tts_model,
    ref_audio_path: Path,
    ref_text: str,
    indicconformer_model,
    whisper_model,
    utmos_scorer,
    audio_dir: Path,
    concurrency_level: int,
    skip_roundtrip: bool,
) -> TTSFileResult:
    """
    Single worker function — one TTS call + full quality metrics.
    Designed to run concurrently with other workers in a ThreadPoolExecutor.

    Thread-safety note: IndicF5 uses PyTorch under the hood. PyTorch
    CUDA operations release the GIL, so multiple threads can submit GPU
    kernels concurrently. The GPU scheduler interleaves them. We do NOT
    share model state between calls — each call is a stateless forward pass
    (no RNN/LSTM hidden states to worry about for this model architecture).
    """
    sent_id = sent["id"]
    text = sent["text"]

    t0 = time.perf_counter()
    audio, latency = run_tts(tts_model, text, ref_audio_path, ref_text)
    # Note: latency measured inside run_tts with perf_counter — wall time
    # including any GPU scheduling delays from concurrent workers.

    gen_dur = audio_duration(audio)
    rtf = latency / gen_dur if gen_dur > 0 else 0.0

    wav_path = audio_dir / f"{sent_id}_c{concurrency_level:02d}_w{worker_id:02d}.wav"
    save_audio(audio, wav_path)

    sil = silence_ratio(audio)
    clip = clipping_ratio(audio)
    mos = estimate_mos(utmos_scorer, audio)

    if not skip_roundtrip:
        stt = round_trip_stt(indicconformer_model, whisper_model, wav_path, text)
    else:
        stt = {
            "indicconformer_hypothesis": "", "indicconformer_cer": -1.0, "indicconformer_wer": -1.0,
            "whisper_hypothesis": "", "whisper_cer": -1.0, "whisper_wer": -1.0,
        }

    return TTSFileResult(
        sentence_id=sent_id,
        input_text=text,
        generated_wav_path=str(wav_path),
        latency_sec=latency,
        generated_audio_sec=gen_dur,
        rtf=rtf,
        silence_ratio=sil,
        clipping_ratio=clip,
        utmos_score=mos,
        indicconformer_hypothesis=stt["indicconformer_hypothesis"],
        indicconformer_cer=stt["indicconformer_cer"],
        indicconformer_wer=stt["indicconformer_wer"],
        whisper_hypothesis=stt["whisper_hypothesis"],
        whisper_cer=stt["whisper_cer"],
        whisper_wer=stt["whisper_wer"],
        run_index=0,
        concurrency_level=concurrency_level,
        worker_id=worker_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# One concurrency level run
# ─────────────────────────────────────────────────────────────────────────────

def run_at_concurrency(
    tts_model,
    indicconformer_model,
    whisper_model,
    utmos_scorer,
    sentences: list[dict],
    ref_audio_path: Path,
    ref_text: str,
    concurrency: int,
    n_repeats: int,
    audio_dir: Path,
    gpu_monitor: GPUMonitor,
    skip_roundtrip: bool,
) -> TTSBatchResult:
    """
    Run all sentences at a given concurrency level, repeated n_repeats times.

    How requests are distributed:
      - sentences list is divided into chunks of size `concurrency`
      - Each chunk is submitted simultaneously to the ThreadPoolExecutor
      - All workers in a chunk start at the same time (synchronized via executor)
      - Wall time measured from chunk start → last worker finishes
      - This simulates a burst of `concurrency` simultaneous users
    """
    all_file_results: list[TTSFileResult] = []
    all_latencies: list[float] = []
    all_wall_times: list[float] = []
    all_vram: list[float] = []
    all_util: list[float] = []

    for repeat_idx in range(n_repeats):
        log.info(f"\n  Repeat {repeat_idx + 1}/{n_repeats}")

        # Rotate sentence order each repeat
        offset = (repeat_idx * concurrency) % max(len(sentences), 1)
        rotated = sentences[offset:] + sentences[:offset]
        chunks = [rotated[i:i+concurrency] for i in range(0, len(rotated), concurrency)]

        repeat_latencies: list[float] = []
        repeat_wall_total = 0.0
        repeat_file_results: list[TTSFileResult] = []

        for chunk_idx, chunk in enumerate(chunks):
            actual_concurrency = len(chunk)  # last chunk may be smaller
            log.info(f"    Chunk {chunk_idx+1}/{len(chunks)} — {actual_concurrency} concurrent workers")

            futures = {}
            gpu_monitor.start()
            chunk_start = time.perf_counter()

            with ThreadPoolExecutor(max_workers=actual_concurrency) as executor:
                for worker_id, sent in enumerate(chunk):
                    fut = executor.submit(
                        _worker_inference,
                        worker_id, sent, tts_model,
                        ref_audio_path, ref_text,
                        indicconformer_model, whisper_model, utmos_scorer,
                        audio_dir, concurrency, skip_roundtrip,
                    )
                    futures[fut] = worker_id

                # Collect results as they complete
                chunk_file_results = []
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        chunk_file_results.append(result)
                        repeat_latencies.append(result.latency_sec)
                    except Exception as e:
                        log.error(f"Worker {futures[fut]} failed: {e}")

            chunk_wall_time = time.perf_counter() - chunk_start
            gpu_stats = gpu_monitor.stop()

            repeat_wall_total += chunk_wall_time
            repeat_file_results.extend(chunk_file_results)
            all_vram.append(gpu_stats["peak_vram_mb"])
            all_util.append(gpu_stats["mean_util_pct"])

            completed = len(chunk_file_results)
            throughput = completed / chunk_wall_time if chunk_wall_time > 0 else 0
            log.info(
                f"    → chunk done: {chunk_wall_time:.3f}s | "
                f"{throughput:.2f} sent/s | "
                f"VRAM={gpu_stats['peak_vram_mb']:.0f}MB | "
                f"GPU={gpu_stats['mean_util_pct']:.1f}%"
            )

        all_file_results.extend(repeat_file_results)
        all_latencies.extend(repeat_latencies)
        all_wall_times.append(repeat_wall_total)

        repeat_tp = len(repeat_file_results) / repeat_wall_total if repeat_wall_total > 0 else 0
        log.info(
            f"  Repeat {repeat_idx+1} total: {repeat_wall_total:.2f}s | "
            f"{repeat_tp:.2f} sent/s | "
            f"mean lat={statistics.mean(repeat_latencies):.3f}s"
        )

    # Aggregate across all repeats
    sorted_lat = sorted(all_latencies)
    n = len(sorted_lat)

    valid_mos = [r.utmos_score for r in all_file_results if r.utmos_score >= 0]
    valid_ic_cer = [r.indicconformer_cer for r in all_file_results if r.indicconformer_cer >= 0]
    valid_w_cer = [r.whisper_cer for r in all_file_results if r.whisper_cer >= 0]
    valid_ic_wer = [r.indicconformer_wer for r in all_file_results if r.indicconformer_wer >= 0]
    valid_w_wer = [r.whisper_wer for r in all_file_results if r.whisper_wer >= 0]

    return TTSBatchResult(
        concurrency_level=concurrency,
        n_sentences=len(sentences),
        total_wall_time_sec=statistics.mean(all_wall_times),
        mean_latency_sec=statistics.mean(all_latencies),
        p50_latency_sec=sorted_lat[n // 2] if n > 0 else 0.0,
        p95_latency_sec=sorted_lat[int(n * 0.95)] if n > 0 else 0.0,
        throughput_sentences_per_sec=len(all_file_results) / sum(all_wall_times) if sum(all_wall_times) > 0 else 0.0,
        mean_rtf=statistics.mean(r.rtf for r in all_file_results),
        mean_silence_ratio=statistics.mean(r.silence_ratio for r in all_file_results),
        mean_clipping_ratio=statistics.mean(r.clipping_ratio for r in all_file_results),
        mean_utmos_score=statistics.mean(valid_mos) if valid_mos else -1.0,
        mean_indicconformer_cer=statistics.mean(valid_ic_cer) if valid_ic_cer else -1.0,
        mean_indicconformer_wer=statistics.mean(valid_ic_wer) if valid_ic_wer else -1.0,
        mean_whisper_cer=statistics.mean(valid_w_cer) if valid_w_cer else -1.0,
        mean_whisper_wer=statistics.mean(valid_w_wer) if valid_w_wer else -1.0,
        gpu_peak_vram_mb=max(all_vram) if all_vram else 0.0,
        gpu_mean_util_pct=statistics.mean(all_util) if all_util else 0.0,
        gpu_peak_util_pct=max(all_util) if all_util else 0.0,
        file_results=all_file_results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(results: list[TTSBatchResult]) -> dict:
    if not results:
        return {}

    best_tp = max(results, key=lambda r: r.throughput_sentences_per_sec)
    baseline = results[0]  # concurrency=1

    # Find the concurrency level where P95 latency exceeds 2× baseline P95
    p95_threshold = baseline.p95_latency_sec * 2.0
    degradation_point = None
    for r in results:
        if r.p95_latency_sec > p95_threshold:
            degradation_point = r.concurrency_level
            break

    # Check quality stability: MOS and CER should not degrade >5% under load
    mos_values = [r.mean_utmos_score for r in results if r.mean_utmos_score >= 0]
    cer_values = [r.mean_indicconformer_cer for r in results if r.mean_indicconformer_cer >= 0]
    mos_stable = (max(mos_values) - min(mos_values)) < 0.2 if len(mos_values) > 1 else True
    cer_stable = (max(cer_values) - min(cer_values)) < 0.05 if len(cer_values) > 1 else True

    return {
        "concurrency_levels_tested": [r.concurrency_level for r in results],
        "baseline_concurrency_1_mean_latency_sec": baseline.mean_latency_sec,
        "baseline_concurrency_1_p95_latency_sec": baseline.p95_latency_sec,
        "best_throughput_concurrency_level": best_tp.concurrency_level,
        "best_throughput_sentences_per_sec": best_tp.throughput_sentences_per_sec,
        "p95_latency_degradation_point_concurrency": degradation_point or "not reached",
        "quality_mos_stable_across_concurrency": mos_stable,
        "quality_cer_stable_across_concurrency": cer_stable,
        "recommended_max_concurrency": (degradation_point - 1) if degradation_point and degradation_point > 1 else best_tp.concurrency_level,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS Phase 2 — Concurrency stress benchmarking")
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--sentences", default="./test_sentences.json")
    p.add_argument("--output", default="./tts_results/phase2")
    p.add_argument("--model", default="ai4bharat/IndicF5")
    p.add_argument("--indicconformer-model", default="ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY_LEVELS,
                   help=f"Concurrency levels to sweep (default: {DEFAULT_CONCURRENCY_LEVELS})")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="Max sentences (smoke test)")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--monitor-interval-ms", type=int, default=250)
    p.add_argument("--skip-roundtrip", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ref_audio = Path(args.ref_audio)
    output_dir = Path(args.output)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not ref_audio.exists():
        log.error(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("TTS BENCHMARK — PHASE 2: CONCURRENCY SWEEP")
    log.info("=" * 60)
    log.info(f"Concurrency levels : {args.concurrency}")
    log.info(f"Repeats per level  : {args.repeats}")

    sentences = load_test_sentences(Path(args.sentences), limit=args.limit)
    if not sentences:
        log.error("No sentences to benchmark.")
        sys.exit(1)

    tts_model = load_indicf5(args.model)
    utmos_scorer = load_utmos()
    indicconformer_model = None
    whisper_model = None
    if not args.skip_roundtrip:
        indicconformer_model = load_indicconformer_stt(args.indicconformer_model)
        whisper_model = load_whisper_stt(args.whisper_model)

    gpu_monitor = GPUMonitor(gpu_index=args.gpu_index, interval_ms=args.monitor_interval_ms)

    # Warm-up
    log.info("\nWarm-up pass...")
    run_tts(tts_model, sentences[0]["text"], ref_audio, args.ref_text)
    log.info("Warm-up complete.\n")

    all_batch_results: list[TTSBatchResult] = []

    for concurrency in args.concurrency:
        log.info(f"\n{'='*60}")
        log.info(f"CONCURRENCY LEVEL: {concurrency}")
        log.info(f"{'='*60}")

        batch_result = run_at_concurrency(
            tts_model, indicconformer_model, whisper_model, utmos_scorer,
            sentences, ref_audio, args.ref_text,
            concurrency=concurrency,
            n_repeats=args.repeats,
            audio_dir=audio_dir,
            gpu_monitor=gpu_monitor,
            skip_roundtrip=args.skip_roundtrip,
        )
        all_batch_results.append(batch_result)

        log.info(f"\n  ▶ concurrency={concurrency} AGGREGATE:")
        log.info(f"    throughput   : {batch_result.throughput_sentences_per_sec:.3f} sent/s")
        log.info(f"    mean latency : {batch_result.mean_latency_sec:.3f}s")
        log.info(f"    P95 latency  : {batch_result.p95_latency_sec:.3f}s")
        log.info(f"    mean MOS     : {batch_result.mean_utmos_score:.2f}")
        log.info(f"    IC-CER       : {batch_result.mean_indicconformer_cer:.3f}")
        log.info(f"    W-CER        : {batch_result.mean_whisper_cer:.3f}")
        log.info(f"    peak VRAM    : {batch_result.gpu_peak_vram_mb:.0f} MB")
        log.info(f"    mean GPU util: {batch_result.gpu_mean_util_pct:.1f}%")

    summary = build_summary(all_batch_results)

    log.info("\n" + "=" * 60)
    log.info("PHASE 2 SUMMARY")
    log.info("=" * 60)
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    report = TTSPhaseReport(
        phase="tts_phase2",
        model_name=args.model,
        reference_audio=str(ref_audio),
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=[asdict(r) for r in all_batch_results],
        summary=summary,
    )
    save_tts_report(report, output_dir)
    log.info(f"\nDone. Results → {output_dir}")


if __name__ == "__main__":
    main()
