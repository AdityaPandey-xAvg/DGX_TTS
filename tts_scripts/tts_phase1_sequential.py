#!/usr/bin/env python3
"""
tts_phase1_sequential.py — TTS Sequential Benchmarking
=======================================================
Sends sentences one by one to IndicF5.
Runs each sentence N times independently (default 5).
Full quality suite per run: latency, RTF, MOS, audio stats, round-trip STT.

Usage:
    python tts_phase1_sequential.py \
        --ref-audio    ./ref_clips/ref_speaker.wav \
        --ref-text     "यह एक संदर्भ वाक्य है जो वक्ता की आवाज़ को दर्शाता है।" \
        --sentences    ./test_sentences.json \
        --output       ./tts_results/phase1 \
        --runs         5

    # Smoke test (3 sentences, 1 run, skip round-trip STT):
    python tts_phase1_sequential.py \
        --ref-audio    ./ref_clips/ref_speaker.wav \
        --ref-text     "reference text" \
        --output       ./tts_results/phase1 \
        --limit        3 \
        --runs         1 \
        --skip-roundtrip
    python tts_phase1_sequential.py \
        --ref-audio    ./ref_clips/ref_speaker.wav \
        --ref-text     "हलो मेरा नाम आदित्य है।" \
        --output       ./tts_results/phase1 \
        --limit        3 \
        --runs         1 \
        --skip-roundtrip

Reference audio requirements:
    - Duration: 3–15 seconds (IndicF5 supports up to ~15s ref; total output ≤ 30s)
    - Format: WAV, 24 kHz mono preferred (model resamples if needed)
    - Content: Clear speech in the target language/voice, minimal background noise
    - The ref-text must be the EXACT transcript of ref-audio — errors here
      degrade voice cloning quality and make the benchmark non-representative

Output:
    tts_phase1_results.json   — full results per sentence per run
    tts_phase1_report.md      — human-readable table
    audio/                    — generated WAV files (sentence_id_run_N.wav)
"""

import argparse
import logging
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tts_benchmark_base import (
    GPUMonitor,
    TTSFileResult,
    TTSPhaseReport,
    audio_duration,
    clipping_ratio,
    compute_audio_quality,
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


# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(
    tts_model,
    indicconformer_model,
    whisper_model,
    utmos_scorer,
    sentences: list[dict],
    ref_audio_path: Path,
    ref_text: str,
    n_runs: int,
    output_dir: Path,
    gpu_monitor: GPUMonitor,
    skip_roundtrip: bool = False,
) -> list[TTSFileResult]:

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[TTSFileResult] = []
    total = len(sentences) * n_runs
    done = 0

    for sent in sentences:
        sent_id = sent["id"]
        text = sent["text"]
        cat = sent.get("length_category", "unknown")

        log.info(f"\n{'─'*60}")
        log.info(f"Sentence [{cat}]: {sent_id}")
        log.info(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        run_latencies = []

        for run_idx in range(n_runs):
            # ── TTS inference (timed + GPU monitored) ────────────────────────
            gpu_monitor.start()
            audio, latency = run_tts(tts_model, text, ref_audio_path, ref_text)
            gpu_stats = gpu_monitor.stop()
            # ─────────────────────────────────────────────────────────────────

            gen_dur = audio_duration(audio)
            rtf = latency / gen_dur if gen_dur > 0 else 0.0
            run_latencies.append(latency)

            # Save audio
            wav_path = audio_dir / f"{sent_id}_run{run_idx:02d}.wav"
            save_audio(audio, wav_path)

            # Audio quality metrics
            sil_ratio = silence_ratio(audio)
            clip_ratio = clipping_ratio(audio)

            # MOS
            mos = estimate_mos(utmos_scorer, audio)

            # Round-trip STT
            if not skip_roundtrip:
                stt_results = round_trip_stt(
                    indicconformer_model, whisper_model, wav_path, text
                )
            else:
                stt_results = {
                    "indicconformer_hypothesis": "", "indicconformer_cer": -1.0, "indicconformer_wer": -1.0,
                    "whisper_hypothesis": "", "whisper_cer": -1.0, "whisper_wer": -1.0,
                }

            result = TTSFileResult(
                sentence_id=sent_id,
                input_text=text,
                generated_wav_path=str(wav_path),
                latency_sec=latency,
                generated_audio_sec=gen_dur,
                rtf=rtf,
                silence_ratio=sil_ratio,
                clipping_ratio=clip_ratio,
                utmos_score=mos,
                indicconformer_hypothesis=stt_results["indicconformer_hypothesis"],
                indicconformer_cer=stt_results["indicconformer_cer"],
                indicconformer_wer=stt_results["indicconformer_wer"],
                whisper_hypothesis=stt_results["whisper_hypothesis"],
                whisper_cer=stt_results["whisper_cer"],
                whisper_wer=stt_results["whisper_wer"],
                run_index=run_idx,
            )
            all_results.append(result)
            done += 1

            log.info(
                f"  Run {run_idx+1}/{n_runs} → "
                f"latency={latency:.3f}s | audio={gen_dur:.2f}s | RTF={rtf:.3f} | "
                f"MOS={mos:.2f} | silence={sil_ratio:.3f} | clip={clip_ratio:.4f} | "
                f"IC-CER={stt_results['indicconformer_cer']:.3f} | W-CER={stt_results['whisper_cer']:.3f} | "
                f"VRAM={gpu_stats['peak_vram_mb']:.0f}MB | GPU={gpu_stats['mean_util_pct']:.1f}% "
                f"[{done}/{total}]"
            )

        mean_lat = statistics.mean(run_latencies)
        std_lat = statistics.stdev(run_latencies) if len(run_latencies) > 1 else 0.0
        log.info(f"  → {sent_id} mean latency: {mean_lat:.3f}s ± {std_lat:.3f}s")

    return all_results


def build_summary(results: list[TTSFileResult]) -> dict:
    latencies = [r.latency_sec for r in results]
    rtfs = [r.rtf for r in results]
    mos_scores = [r.utmos_score for r in results if r.utmos_score >= 0]
    ic_cers = [r.indicconformer_cer for r in results if r.indicconformer_cer >= 0]
    w_cers = [r.whisper_cer for r in results if r.whisper_cer >= 0]
    sil = [r.silence_ratio for r in results]
    clip = [r.clipping_ratio for r in results]

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "n_sentences": len(set(r.sentence_id for r in results)),
        "n_runs_per_sentence": len(set(r.run_index for r in results)),
        "total_inference_calls": len(results),
        "mean_latency_sec": statistics.mean(latencies),
        "std_latency_sec": statistics.stdev(latencies) if n > 1 else 0.0,
        "p50_latency_sec": sorted_lat[n // 2],
        "p95_latency_sec": sorted_lat[int(n * 0.95)],
        "p99_latency_sec": sorted_lat[int(n * 0.99)],
        "mean_rtf": statistics.mean(rtfs),
        "mean_utmos_mos": statistics.mean(mos_scores) if mos_scores else -1.0,
        "mean_silence_ratio": statistics.mean(sil),
        "mean_clipping_ratio": statistics.mean(clip),
        "mean_indicconformer_cer": statistics.mean(ic_cers) if ic_cers else -1.0,
        "mean_indicconformer_wer": statistics.mean([r.indicconformer_wer for r in results if r.indicconformer_wer >= 0]) if ic_cers else -1.0,
        "mean_whisper_cer": statistics.mean(w_cers) if w_cers else -1.0,
        "mean_whisper_wer": statistics.mean([r.whisper_wer for r in results if r.whisper_wer >= 0]) if w_cers else -1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS Phase 1 — Sequential benchmarking")
    p.add_argument("--ref-audio", required=True, help="Path to reference speaker WAV (3–15 seconds)")
    p.add_argument("--ref-text", required=True, help="Exact transcript of the reference audio")
    p.add_argument("--sentences", default="./test_sentences.json", help="Path to test sentences JSON")
    p.add_argument("--output", default="./tts_results/phase1")
    p.add_argument("--model", default="ai4bharat/IndicF5")
    p.add_argument("--indicconformer-model", default="ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="Max sentences (smoke test)")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--monitor-interval-ms", type=int, default=250)
    p.add_argument("--skip-roundtrip", action="store_true",
                   help="Skip round-trip STT (faster runs, no CER/WER metrics)")
    return p.parse_args()


def main():
    args = parse_args()
    ref_audio = Path(args.ref_audio)
    output_dir = Path(args.output)
    sentences_path = Path(args.sentences)

    if not ref_audio.exists():
        log.error(f"Reference audio not found: {ref_audio}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("TTS BENCHMARK — PHASE 1: SEQUENTIAL")
    log.info("=" * 60)
    log.info(f"Reference audio : {ref_audio}")
    log.info(f"Reference text  : {args.ref_text[:60]}...")

    sentences = load_test_sentences(sentences_path, limit=args.limit)
    if not sentences:
        log.error("No sentences to benchmark. Exiting.")
        sys.exit(1)

    # Load models
    tts_model = load_indicf5(args.model)
    utmos_scorer = load_utmos()

    indicconformer_model = None
    whisper_model = None
    if not args.skip_roundtrip:
        indicconformer_model = load_indicconformer_stt(args.indicconformer_model)
        whisper_model = load_whisper_stt(args.whisper_model)

    gpu_monitor = GPUMonitor(gpu_index=args.gpu_index, interval_ms=args.monitor_interval_ms)

    # Warm-up (not recorded)
    log.info("\nWarm-up pass...")
    run_tts(tts_model, sentences[0]["text"], ref_audio, args.ref_text)
    log.info("Warm-up complete.\n")

    log.info(f"Running: {len(sentences)} sentences × {args.runs} runs = {len(sentences)*args.runs} calls\n")

    results = run_phase1(
        tts_model, indicconformer_model, whisper_model, utmos_scorer,
        sentences, ref_audio, args.ref_text,
        n_runs=args.runs, output_dir=output_dir,
        gpu_monitor=gpu_monitor, skip_roundtrip=args.skip_roundtrip,
    )

    summary = build_summary(results)

    log.info("\n" + "=" * 60)
    log.info("PHASE 1 SUMMARY")
    log.info("=" * 60)
    for k, v in summary.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report = TTSPhaseReport(
        phase="tts_phase1",
        model_name=args.model,
        reference_audio=str(ref_audio),
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=[asdict(r) for r in results],
        summary=summary,
    )
    save_tts_report(report, output_dir)
    log.info(f"\nDone. Results → {output_dir}")


if __name__ == "__main__":
    main()
