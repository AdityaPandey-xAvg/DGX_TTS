#!/usr/bin/env python3
"""
tts_phase2_concurrent.py — TTS Concurrency Stress Benchmarking
===============================================================
Same logic as before, with one addition:
  Every thread now logs a detailed CSV row with:
    - concurrency_level, loop_number, total_sentences_in_batch
    - thread_id, sentence_id, sentence_text
    - thread_start_time_sec (seconds since chunk start, e.g. 0.001 = thread started 1ms after chunk fired)
    - thread_end_time_sec   (seconds since chunk start, e.g. 2.341 = thread finished 2.341s after chunk fired)
    - duration_sec      (end - start, in seconds)
    - generated_audio_sec, rtf
    - gpu_vram_mb, gpu_util_pct  (sampled per-thread via pynvml)

CSV output: tts_phase2_thread_log.csv
  One row per thread per sentence per loop per concurrency level.

  Example for 8 sentences, 2 repeats, concurrency [1,2,4,8]:
    Concurrency 1 → 16 rows  (1 thread × 8 sentences × 2 loops)
    Concurrency 2 → 16 rows  (2 threads × 4 chunks   × 2 loops)
    Concurrency 4 → 16 rows  (4 threads × 2 chunks   × 2 loops)
    Concurrency 8 → 16 rows  (8 threads × 1 chunk    × 2 loops)
    Total         → 64 rows

Usage:
    python tts_phase2_concurrent.py \
        --ref-audio   ./my_hindi.wav \
        --ref-text    "transcript" \
        --sentences   ./concurrency_test.json \
        --output      ./tts_results/phase2 \
        --concurrency 1 2 4 8 \
        --repeats     2 \
        --skip-roundtrip
"""

import argparse
import csv
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "concurrency_level",   # number of threads (1, 2, 4, 8)
    "loop_number",         # which repeat (1-based)
    "total_sentences",     # total sentences in this batch
    "thread_id",           # which thread (0-based)
    "thread_start_sec",    # seconds since benchmark started when thread began
    "thread_end_sec",      # seconds since benchmark started when thread completed
    "total_duration_sec",  # thread_end_sec - thread_start_sec
    "gpu_vram_mb",         # GPU VRAM at thread completion
    "gpu_util_pct",        # GPU utilisation % at thread completion
]


def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_csv_row(csv_path: Path, row: dict):
    """Append one row to the CSV. Thread-safe via file append mode."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# GPU snapshot — reads current VRAM + utilisation instantly (no background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_snapshot(gpu_index: int = 0) -> tuple[float, float]:
    """
    Returns (vram_mb, util_pct) at this exact moment.
    Returns (0.0, 0.0) if pynvml is unavailable.
    Called once per thread right after inference completes.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return mem.used / 1024 / 1024, float(util.gpu)
    except Exception:
        return 0.0, 0.0


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
    loop_number: int,
    total_sentences_batch: int,
    gpu_index: int,
    csv_path: Path,
    skip_roundtrip: bool,
    benchmark_t0: float = 0.0,   # perf_counter value when the full benchmark started
) -> TTSFileResult:
    """
    Single worker — runs in its own thread.

    Timeline inside this function:
      thread_start_sec  ← seconds since benchmark_t0, recorded first thing
            |
            |  model(text, ref_audio, ref_text)  ← thread blocked here
            |  GPU does the work (interleaved with other threads)
            |
      thread_end_sec    ← seconds since benchmark_t0, recorded when model() returns
      total_duration_sec = thread_end_sec - thread_start_sec

    Each thread measures its own start/end independently, so you can see
    exactly when each one started and finished relative to each other.
    """
    sent_id = sent["id"]
    text = sent["text"]

    # perf_counter relative to benchmark start — so row 1 starts at e.g. 0.0012,
    # row 2 starts at 2.3412 (right after row 1 ended), and so on.
    t_start = time.perf_counter()
    thread_start_sec = round(t_start - benchmark_t0, 4)

    # ── Model call — thread blocks here until GPU returns audio ───────────────
    audio, _ = run_tts(tts_model, text, ref_audio_path, ref_text)

    # ── Record end immediately after model returns ────────────────────────────
    t_end = time.perf_counter()
    thread_end_sec   = round(t_end - benchmark_t0, 4)
    total_duration_sec = round(t_end - t_start, 4)

    # ── GPU snapshot — read current GPU state right after inference ───────────
    gpu_vram_mb, gpu_util_pct = _gpu_snapshot(gpu_index)

    # ── Audio metrics ─────────────────────────────────────────────────────────
    gen_dur = audio_duration(audio)
    rtf = duration_sec / gen_dur if gen_dur > 0 else 0.0
    sil = silence_ratio(audio)
    clip = clipping_ratio(audio)
    mos = estimate_mos(utmos_scorer, audio)

    wav_path = audio_dir / f"{sent_id}_c{concurrency_level:02d}_w{worker_id:02d}_l{loop_number:02d}.wav"
    save_audio(audio, wav_path)

    # ── Round-trip STT ────────────────────────────────────────────────────────
    if not skip_roundtrip:
        stt = round_trip_stt(indicconformer_model, whisper_model, wav_path, text)
    else:
        stt = {
            "indicconformer_hypothesis": "", "indicconformer_cer": -1.0, "indicconformer_wer": -1.0,
            "whisper_hypothesis": "", "whisper_cer": -1.0, "whisper_wer": -1.0,
        }

    # ── Write CSV row ────────────────────────────────────────────────────────
    append_csv_row(csv_path, {
        "concurrency_level":  concurrency_level,
        "loop_number":        loop_number,
        "total_sentences":    total_sentences_batch,
        "thread_id":          worker_id,
        "thread_start_sec":   thread_start_sec,
        "thread_end_sec":     thread_end_sec,
        "total_duration_sec": total_duration_sec,
        "gpu_vram_mb":        round(gpu_vram_mb, 1),
        "gpu_util_pct":       round(gpu_util_pct, 1),
    })

    log.info(
        f"    [c={concurrency_level} loop={loop_number} thread={worker_id}] "
        f"start={thread_start_sec}s → end={thread_end_sec}s | "
        f"duration={total_duration_sec}s | "
        f"VRAM={gpu_vram_mb:.0f}MB | GPU={gpu_util_pct:.1f}%"
    )

    return TTSFileResult(
        sentence_id=sent_id,
        input_text=text,
        generated_wav_path=str(wav_path),
        latency_sec=total_duration_sec,
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
        run_index=loop_number - 1,
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
    gpu_index: int,
    csv_path: Path,
    skip_roundtrip: bool,
    benchmark_t0: float = 0.0,
) -> TTSBatchResult:
    """
    Run all sentences at a given concurrency level, repeated n_repeats times.

    Structure:
      For each repeat (loop):
        Split sentences into chunks of size `concurrency`
        For each chunk:
          Fire `concurrency` threads simultaneously
          Each thread handles one sentence
          All threads start at the same moment
          Wait for all threads to finish
          Move to next chunk
    """
    all_file_results: list[TTSFileResult] = []
    all_latencies: list[float] = []
    all_wall_times: list[float] = []
    all_vram: list[float] = []
    all_util: list[float] = []

    for repeat_idx in range(n_repeats):
        loop_number = repeat_idx + 1
        log.info(f"\n  Loop {loop_number}/{n_repeats}")

        # Rotate sentence order each repeat to avoid positional bias
        offset = (repeat_idx * concurrency) % max(len(sentences), 1)
        rotated = sentences[offset:] + sentences[:offset]
        chunks = [rotated[i:i + concurrency] for i in range(0, len(rotated), concurrency)]

        repeat_latencies: list[float] = []
        repeat_wall_total = 0.0
        repeat_file_results: list[TTSFileResult] = []

        for chunk_idx, chunk in enumerate(chunks):
            actual_concurrency = len(chunk)
            log.info(
                f"    Chunk {chunk_idx+1}/{len(chunks)} — "
                f"{actual_concurrency} threads firing simultaneously"
            )

            gpu_monitor.start()
            chunk_start = time.perf_counter()

            # ── Fire all threads simultaneously ───────────────────────────────
            with ThreadPoolExecutor(max_workers=actual_concurrency) as executor:
                futures = {
                    executor.submit(
                        _worker_inference,
                        worker_id=i,
                        sent=chunk[i],
                        tts_model=tts_model,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        indicconformer_model=indicconformer_model,
                        whisper_model=whisper_model,
                        utmos_scorer=utmos_scorer,
                        audio_dir=audio_dir,
                        concurrency_level=concurrency,
                        loop_number=loop_number,
                        total_sentences_batch=actual_concurrency,
                        gpu_index=gpu_index,
                        csv_path=csv_path,
                        skip_roundtrip=skip_roundtrip,
                        benchmark_t0=benchmark_t0,
                    ): i
                    for i in range(actual_concurrency)
                }

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

            log.info(
                f"    → chunk wall time: {chunk_wall_time:.3f}s | "
                f"VRAM={gpu_stats['peak_vram_mb']:.0f}MB | "
                f"GPU={gpu_stats['mean_util_pct']:.1f}%"
            )

        all_file_results.extend(repeat_file_results)
        all_latencies.extend(repeat_latencies)
        all_wall_times.append(repeat_wall_total)

        repeat_tp = len(repeat_file_results) / repeat_wall_total if repeat_wall_total > 0 else 0
        log.info(
            f"  Loop {loop_number} done: {repeat_wall_total:.2f}s total | "
            f"{repeat_tp:.2f} sent/s | "
            f"mean thread duration={statistics.mean(repeat_latencies):.3f}s"
        )

    # ── Aggregate across all loops ─────────────────────────────────────────
    sorted_lat = sorted(all_latencies)
    n = len(sorted_lat)

    valid_mos    = [r.utmos_score for r in all_file_results if r.utmos_score >= 0]
    valid_ic_cer = [r.indicconformer_cer for r in all_file_results if r.indicconformer_cer >= 0]
    valid_w_cer  = [r.whisper_cer for r in all_file_results if r.whisper_cer >= 0]
    valid_ic_wer = [r.indicconformer_wer for r in all_file_results if r.indicconformer_wer >= 0]
    valid_w_wer  = [r.whisper_wer for r in all_file_results if r.whisper_wer >= 0]

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

    best_tp  = max(results, key=lambda r: r.throughput_sentences_per_sec)
    baseline = results[0]

    p95_threshold    = baseline.p95_latency_sec * 2.0
    degradation_point = None
    for r in results:
        if r.p95_latency_sec > p95_threshold:
            degradation_point = r.concurrency_level
            break

    mos_values = [r.mean_utmos_score for r in results if r.mean_utmos_score >= 0]
    cer_values = [r.mean_indicconformer_cer for r in results if r.mean_indicconformer_cer >= 0]
    mos_stable = (max(mos_values) - min(mos_values)) < 0.2 if len(mos_values) > 1 else True
    cer_stable = (max(cer_values) - min(cer_values)) < 0.05 if len(cer_values) > 1 else True

    return {
        "concurrency_levels_tested":              [r.concurrency_level for r in results],
        "baseline_concurrency_1_mean_latency_sec": baseline.mean_latency_sec,
        "baseline_concurrency_1_p95_latency_sec":  baseline.p95_latency_sec,
        "best_throughput_concurrency_level":       best_tp.concurrency_level,
        "best_throughput_sentences_per_sec":       best_tp.throughput_sentences_per_sec,
        "p95_latency_degradation_point_concurrency": degradation_point or "not reached",
        "quality_mos_stable_across_concurrency":   mos_stable,
        "quality_cer_stable_across_concurrency":   cer_stable,
        "recommended_max_concurrency": (
            (degradation_point - 1)
            if degradation_point and degradation_point > 1
            else best_tp.concurrency_level
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS Phase 2 — Concurrency stress benchmarking")
    p.add_argument("--ref-audio",  required=True)
    p.add_argument("--ref-text",   required=True)
    p.add_argument("--sentences",  default="./test_sentences.json")
    p.add_argument("--output",     default="./tts_results/phase2")
    p.add_argument("--model",      default="ai4bharat/IndicF5")
    p.add_argument("--indicconformer-model", default="ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY_LEVELS)
    p.add_argument("--repeats",     type=int, default=3)
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--gpu-index",   type=int, default=0)
    p.add_argument("--monitor-interval-ms", type=int, default=250)
    p.add_argument("--skip-roundtrip", action="store_true")
    return p.parse_args()


def main():
    args      = parse_args()
    ref_audio = Path(args.ref_audio)
    output_dir = Path(args.output)
    audio_dir  = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not ref_audio.exists():
        log.error(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    # CSV written incrementally — one row appended as soon as each thread finishes
    csv_path = output_dir / "tts_phase2_thread_log.csv"
    write_csv_header(csv_path)
    log.info(f"Thread log CSV → {csv_path}")

    log.info("=" * 60)
    log.info("TTS BENCHMARK — PHASE 2: CONCURRENCY SWEEP")
    log.info("=" * 60)
    log.info(f"Concurrency levels : {args.concurrency}")
    log.info(f"Repeats (loops)    : {args.repeats}")

    sentences = load_test_sentences(Path(args.sentences), limit=args.limit)
    if not sentences:
        log.error("No sentences to benchmark.")
        sys.exit(1)

    tts_model    = load_indicf5(args.model)
    utmos_scorer = load_utmos()
    indicconformer_model = None
    whisper_model        = None
    if not args.skip_roundtrip:
        indicconformer_model = load_indicconformer_stt(args.indicconformer_model)
        whisper_model        = load_whisper_stt(args.whisper_model)

    gpu_monitor = GPUMonitor(gpu_index=args.gpu_index, interval_ms=args.monitor_interval_ms)

    # Warm-up — not logged to CSV
    log.info("\nWarm-up pass (not logged)...")
    run_tts(tts_model, sentences[0]["text"], ref_audio, args.ref_text)
    log.info("Warm-up done.\n")

    # Single zero point for the entire benchmark.
    # thread_start_sec / thread_end_sec in every CSV row is relative to this.
    # Thread 1 starts at ~0.001s, ends at 2.345s.
    # Thread 2 starts at 2.412s, ends at 4.891s. And so on continuously.
    benchmark_t0 = time.perf_counter()

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
            gpu_index=args.gpu_index,
            csv_path=csv_path,
            skip_roundtrip=args.skip_roundtrip,
            benchmark_t0=benchmark_t0,
        )
        all_batch_results.append(batch_result)

        log.info(f"\n  ▶ concurrency={concurrency} AGGREGATE:")
        log.info(f"    throughput   : {batch_result.throughput_sentences_per_sec:.3f} sent/s")
        log.info(f"    mean duration: {batch_result.mean_latency_sec:.3f}s")
        log.info(f"    P95 duration : {batch_result.p95_latency_sec:.3f}s")
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
        timestamp=f"run_{int(time.perf_counter())}",
        results=[asdict(r) for r in all_batch_results],
        summary=summary,
    )
    save_tts_report(report, output_dir)
    log.info(f"\nDone. Results → {output_dir}")
    log.info(f"Thread log    → {csv_path}")


if __name__ == "__main__":
    main()
