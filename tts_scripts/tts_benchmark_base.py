#!/usr/bin/env python3
"""
tts_benchmark_base.py — Shared utilities for TTS benchmarking (Phase 1 & 2)
============================================================================
Provides:
  - IndicF5 model loader
  - Audio quality metrics: silence ratio, clipping rate, duration
  - MOS estimation via UTMOS
  - Round-trip STT quality check (IndicConformer + Whisper) → CER + WER
  - GPU monitor (pynvml)
  - Result dataclasses + JSON/Markdown report writer

Architecture note — why TTS batching differs from STT batching:
  IndicF5 (F5-TTS based) does not expose a native batch forward pass in its
  public inference API. Each call to model(text, ref_audio, ref_text) runs
  one complete flow-matching denoising loop independently. Phase 2 therefore
  implements "concurrency batching" — N simultaneous calls via
  ThreadPoolExecutor — which is the realistic production model (N workers,
  each handling one request). This is distinct from STT batching where one
  GPU forward pass handles B inputs simultaneously.

Do not run directly — import from phase1 / phase2 scripts.
"""

import json
import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

INDICF5_REPO   = "ai4bharat/IndicF5"
TTS_SAMPLE_RATE = 24_000   # IndicF5 output sample rate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Result schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTSFileResult:
    """Per-sentence result for one TTS inference call."""
    sentence_id: str
    input_text: str
    generated_wav_path: str          # where we saved the output audio
    # ── Timing ──────────────────────────────────────────────────────────────
    latency_sec: float               # wall time from call → audio tensor returned
    generated_audio_sec: float       # duration of the generated audio
    rtf: float                       # latency / generated_audio_sec  (<1 = faster than real-time)
    # ── Audio quality ────────────────────────────────────────────────────────
    silence_ratio: float             # fraction of samples below silence threshold (0–1)
    clipping_ratio: float            # fraction of samples at ±1.0 (0–1)
    # ── MOS (UTMOS automated scorer) ─────────────────────────────────────────
    utmos_score: float               # predicted MOS 1–5 (-1 if scorer unavailable)
    # ── Round-trip STT quality ───────────────────────────────────────────────
    indicconformer_hypothesis: str
    indicconformer_cer: float        # Character Error Rate vs input_text
    indicconformer_wer: float        # Word Error Rate vs input_text
    whisper_hypothesis: str
    whisper_cer: float
    whisper_wer: float
    # ── Context ──────────────────────────────────────────────────────────────
    run_index: int
    concurrency_level: Optional[int] = None   # Phase 2: how many concurrent calls
    worker_id: Optional[int] = None           # Phase 2: which worker produced this


@dataclass
class TTSBatchResult:
    """Aggregate result for one concurrency level (Phase 2)."""
    concurrency_level: int
    n_sentences: int
    total_wall_time_sec: float          # time from first call start → last call end
    mean_latency_sec: float
    p50_latency_sec: float
    p95_latency_sec: float
    throughput_sentences_per_sec: float
    mean_rtf: float
    mean_silence_ratio: float
    mean_clipping_ratio: float
    mean_utmos_score: float
    mean_indicconformer_cer: float
    mean_indicconformer_wer: float
    mean_whisper_cer: float
    mean_whisper_wer: float
    gpu_peak_vram_mb: float
    gpu_mean_util_pct: float
    gpu_peak_util_pct: float
    file_results: list[TTSFileResult] = field(default_factory=list)


@dataclass
class TTSPhaseReport:
    phase: str
    model_name: str
    reference_audio: str
    timestamp: str
    results: list
    summary: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_indicf5(repo_id: str = INDICF5_REPO):
    """
    Load IndicF5 from HuggingFace and explicitly move to CUDA if available.

    Requires:
        pip install git+https://github.com/ai4bharat/IndicF5.git
        huggingface-cli login   (model is gated — accept terms on HF first)

    The model is a flow-matching TTS. Each inference call:
        audio_array = model(text, ref_audio_path, ref_text)
    Returns a numpy float32 array at 24 kHz.

    Device placement note:
        AutoModel.from_pretrained() without device_map loads onto CPU by
        default regardless of CUDA availability. We must call .to(device)
        explicitly after loading — this is the pattern used in AI4Bharat's
        own official Gradio Space (app.py). Without this, all inference
        runs on CPU and RTF will be ~17x instead of <1x on GPU.

    transformers version note:
        Pin transformers==4.49.0 to avoid the meta-tensor error on newer
        versions. Run: pip install transformers==4.49.0
    """
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers==4.49.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading IndicF5 from: {repo_id}")
    log.info(f"Target device: {device}")
    if device.type == "cpu":
        log.warning(
            "CUDA not available — IndicF5 will run on CPU. "
            "Expect RTF ~15–20x (very slow). "
            "Ensure PyTorch is installed with CUDA support: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    # Explicitly move to device — AutoModel.from_pretrained does NOT do this
    # automatically without device_map="auto". This is the root cause of the
    # CPU-speed issue: the model loads but stays on CPU until .to(device).
    model = model.to(device)

    elapsed = time.perf_counter() - t0

    # Confirm actual device of model parameters
    try:
        actual_device = next(model.parameters()).device
        log.info(f"IndicF5 loaded in {elapsed:.1f}s — parameters on: {actual_device}")
        if actual_device.type == "cpu" and torch.cuda.is_available():
            log.warning(
                "Model parameters are on CPU despite CUDA being available. "
                "This may indicate a meta-tensor loading issue. "
                "Try: pip install transformers==4.49.0 and reload."
            )
    except StopIteration:
        log.info(f"IndicF5 loaded in {elapsed:.1f}s")

    return model


def load_indicconformer_stt(model_name: str = "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"):
    """Load IndicConformer for round-trip STT quality check."""
    try:
        import nemo.collections.asr as nemo_asr
        import torch
    except ImportError:
        raise ImportError("NeMo not installed. Run: pip install nemo_toolkit[asr] --break-system-packages")
    log.info(f"Loading IndicConformer (round-trip STT): {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()
    if __import__("torch").cuda.is_available():
        model = model.cuda()
    return model


def load_whisper_stt(model_size: str = "large-v3"):
    """Load Whisper for round-trip STT quality check."""
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed. Run: pip install openai-whisper --break-system-packages")
    log.info(f"Loading Whisper {model_size} (round-trip STT)")
    model = whisper.load_model(model_size)
    return model


def load_utmos():
    """
    Load UTMOS22 MOS predictor.
    Returns scorer object or None if unavailable (non-fatal).

    UTMOS predicts Mean Opinion Score (1–5) from audio without a reference.
    Paper: https://arxiv.org/abs/2204.02152
    """
    try:
        import utmos
        scorer = utmos.SOSFMOSPredictor()
        log.info("UTMOS MOS scorer loaded.")
        return scorer
    except ImportError:
        log.warning(
            "UTMOS not installed — MOS scores will be -1. "
            "Install with: pip install utmos --break-system-packages"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TTS inference
# ─────────────────────────────────────────────────────────────────────────────

def run_tts(model, text: str, ref_audio_path: Path, ref_text: str) -> tuple[np.ndarray, float]:
    """
    Run one IndicF5 inference call.
    Returns (audio_array_float32, latency_sec).

    audio_array is at TTS_SAMPLE_RATE (24 kHz), float32, values in [-1, 1].
    """
    import torch

    # Log GPU memory before call to confirm GPU is being used.
    # If allocated_mb stays near 0 across all calls, model is on CPU.
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
        log.debug(f"GPU memory before inference: {allocated_mb:.0f} MB")

    t0 = time.perf_counter()
    audio = model(text, str(ref_audio_path), ref_text)
    latency = time.perf_counter() - t0

    if torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated() / 1024 / 1024
        log.debug(f"GPU memory after inference: {allocated_after:.0f} MB")
        if allocated_after < 100:
            log.warning(
                f"GPU memory after inference only {allocated_after:.0f} MB — "
                "model may still be on CPU. Check load_indicf5() .to(device) completed."
            )

    # Normalise to float32 in [-1, 1] regardless of what model returns
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) > 0 and (audio.max() > 1.0 or audio.min() < -1.0):
        peak = max(abs(audio.max()), abs(audio.min()))
        if peak > 0:
            audio = audio / peak

    return audio, latency


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = TTS_SAMPLE_RATE):
    """Save audio array to WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Audio quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def audio_duration(audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE) -> float:
    return len(audio) / sample_rate


def silence_ratio(audio: np.ndarray, threshold_db: float = -40.0) -> float:
    """
    Fraction of samples below threshold_db relative to peak.
    High silence ratio (>0.3) in a short utterance indicates generation failure
    (model generated padding/silence instead of speech).
    """
    if len(audio) == 0:
        return 1.0
    peak = np.max(np.abs(audio))
    if peak == 0:
        return 1.0
    threshold_linear = peak * (10 ** (threshold_db / 20))
    return float(np.mean(np.abs(audio) < threshold_linear))


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    """
    Fraction of samples at or above threshold (clipping indicator).
    Values >0.01 suggest the output was improperly normalised or the model
    produced distorted audio.
    """
    return float(np.mean(np.abs(audio) >= threshold))


def compute_audio_quality(audio: np.ndarray) -> dict:
    return {
        "duration_sec": audio_duration(audio),
        "silence_ratio": silence_ratio(audio),
        "clipping_ratio": clipping_ratio(audio),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MOS estimation (UTMOS)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_mos(utmos_scorer, audio: np.ndarray, sample_rate: int = TTS_SAMPLE_RATE) -> float:
    """
    Predict MOS score (1–5) using UTMOS.
    Returns -1.0 if scorer is unavailable.

    Interpretation:
        > 4.0  : near-human quality
        3.5–4.0: good, acceptable for most applications
        3.0–3.5: intelligible but noticeable artefacts
        < 3.0  : poor quality, likely generation failure
    """
    if utmos_scorer is None:
        return -1.0
    try:
        import torch
        tensor = torch.FloatTensor(audio).unsqueeze(0)
        score = utmos_scorer.score(tensor, sampling_rate=sample_rate)
        return float(score)
    except Exception as e:
        log.warning(f"UTMOS scoring failed: {e}")
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Round-trip STT quality (text → TTS → STT → compare)
# ─────────────────────────────────────────────────────────────────────────────

def round_trip_stt(
    indicconformer_model,
    whisper_model,
    wav_path: Path,
    reference_text: str,
) -> dict:
    """
    Transcribe the generated TTS audio with both STT models.
    Compare transcriptions against the original input text to measure
    intelligibility (not naturalness — that's UTMOS's job).

    Returns dict with keys:
        indicconformer_hypothesis, indicconformer_cer, indicconformer_wer
        whisper_hypothesis, whisper_cer, whisper_wer
    """
    result = {
        "indicconformer_hypothesis": "",
        "indicconformer_cer": 1.0,
        "indicconformer_wer": 1.0,
        "whisper_hypothesis": "",
        "whisper_cer": 1.0,
        "whisper_wer": 1.0,
    }

    # IndicConformer
    if indicconformer_model is not None:
        try:
            hypotheses = indicconformer_model.transcribe([str(wav_path)])
            hyp = hypotheses[0].text if hasattr(hypotheses[0], "text") else str(hypotheses[0])
            result["indicconformer_hypothesis"] = hyp
            result["indicconformer_cer"] = _cer(reference_text, hyp)
            result["indicconformer_wer"] = _wer(reference_text, hyp)
        except Exception as e:
            log.warning(f"IndicConformer round-trip failed for {wav_path.name}: {e}")

    # Whisper
    if whisper_model is not None:
        try:
            import whisper
            out = whisper_model.transcribe(str(wav_path), language="hi")
            hyp = out["text"].strip()
            result["whisper_hypothesis"] = hyp
            result["whisper_cer"] = _cer(reference_text, hyp)
            result["whisper_wer"] = _wer(reference_text, hyp)
        except Exception as e:
            log.warning(f"Whisper round-trip failed for {wav_path.name}: {e}")

    return result


def _wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0
    try:
        from jiwer import wer
        return float(wer(reference.strip(), hypothesis.strip()))
    except ImportError:
        return _simple_wer(reference, hypothesis)


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — critical for Indian languages where word boundaries
    are less reliable than character-level alignment."""
    if not reference.strip():
        return 0.0
    try:
        from jiwer import cer
        return float(cer(reference.strip(), hypothesis.strip()))
    except ImportError:
        return _simple_cer(reference, hypothesis)


def _simple_wer(ref: str, hyp: str) -> float:
    """Fallback WER without jiwer."""
    r, h = ref.strip().split(), hyp.strip().split()
    if not r:
        return 0.0
    # Levenshtein on word lists
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)


def _simple_cer(ref: str, hyp: str) -> float:
    """Fallback CER without jiwer."""
    r, h = list(ref.strip()), list(hyp.strip())
    if not r:
        return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GPU monitor  (identical to STT suite)
# ─────────────────────────────────────────────────────────────────────────────

class GPUMonitor:
    """Background thread that samples VRAM + GPU utilisation via pynvml."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 250):
        self.gpu_index = gpu_index
        self.interval_sec = interval_ms / 1000
        self._samples_vram: list[float] = []
        self._samples_util: list[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        self._handle = None
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_available = True
        except Exception as e:
            log.warning(f"GPU monitoring unavailable: {e}")

    def _sample_loop(self):
        import pynvml
        while not self._stop_event.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._samples_vram.append(mem.used / 1024 / 1024)
                self._samples_util.append(float(util.gpu))
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        if not self._nvml_available:
            return
        self._samples_vram.clear()
        self._samples_util.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self._nvml_available or not self._thread:
            return {"peak_vram_mb": 0.0, "mean_util_pct": 0.0, "peak_util_pct": 0.0}
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if not self._samples_vram:
            return {"peak_vram_mb": 0.0, "mean_util_pct": 0.0, "peak_util_pct": 0.0}
        return {
            "peak_vram_mb": max(self._samples_vram),
            "mean_util_pct": statistics.mean(self._samples_util),
            "peak_util_pct": max(self._samples_util),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Test sentence loader
# ─────────────────────────────────────────────────────────────────────────────

def load_test_sentences(sentences_json: Path, limit: Optional[int] = None) -> list[dict]:
    """
    Load curated Hindi/Hinglish test sentences.

    Expected JSON format (list of objects):
        [
          {
            "id": "s001",
            "text": "नमस्ते, आप कैसे हैं?",
            "length_category": "short"   // optional: short / medium / long
          },
          ...
        ]

    If sentences_json does not exist, falls back to a built-in default set
    so the benchmark can run immediately without a custom file.
    """
    if sentences_json.exists():
        with open(sentences_json, encoding="utf-8") as f:
            sentences = json.load(f)
        log.info(f"Loaded {len(sentences)} test sentences from {sentences_json}")
    else:
        log.warning(
            f"Sentence file not found: {sentences_json}\n"
            "Falling back to built-in default Hindi/Hinglish sentences.\n"
            "Create your own file at that path to use custom sentences."
        )
        sentences = _default_sentences()

    if limit:
        sentences = sentences[:limit]
    return sentences


def _default_sentences() -> list[dict]:
    """
    25 built-in Hindi/Hinglish sentences spanning short, medium, and long
    to cover different generation lengths out of the box.
    Replace with your curated set via --sentences flag.
    """
    return [
        # Short (< 10 words)
        {"id": "s001", "text": "नमस्ते, आप कैसे हैं?", "length_category": "short"},
        {"id": "s002", "text": "आज मौसम बहुत अच्छा है।", "length_category": "short"},
        {"id": "s003", "text": "मुझे पानी पीना है।", "length_category": "short"},
        {"id": "s004", "text": "Kya aap meri madad kar sakte hain?", "length_category": "short"},
        {"id": "s005", "text": "बाज़ार कितनी दूर है?", "length_category": "short"},
        {"id": "s006", "text": "यह बहुत सुंदर जगह है।", "length_category": "short"},
        {"id": "s007", "text": "Mujhe thodi der baad milna hai.", "length_category": "short"},
        {"id": "s008", "text": "खाना बहुत स्वादिष्ट था।", "length_category": "short"},
        # Medium (10–25 words)
        {"id": "s009", "text": "कल मैं अपने दोस्त से मिलने जा रहा हूं, हम साथ में खाना खाएंगे।", "length_category": "medium"},
        {"id": "s010", "text": "आज का दिन बहुत व्यस्त रहा, लेकिन मैंने सब काम पूरा कर लिया।", "length_category": "medium"},
        {"id": "s011", "text": "Mere ghar ke paas ek naya restaurant khula hai, wahan ka khana bahut accha hai.", "length_category": "medium"},
        {"id": "s012", "text": "भारत में बहुत सारी भाषाएं बोली जाती हैं, और हर भाषा की अपनी विशेषता है।", "length_category": "medium"},
        {"id": "s013", "text": "इस साल गर्मी बहुत ज्यादा पड़ रही है, लोगों को बाहर निकलने में दिक्कत हो रही है।", "length_category": "medium"},
        {"id": "s014", "text": "Aap ne jo kaam kiya hai, woh bahut badhiya hai, ismein aur sudhar ho sakta hai.", "length_category": "medium"},
        {"id": "s015", "text": "मेरे परिवार में पांच लोग हैं, माता-पिता, दो भाई, और मैं।", "length_category": "medium"},
        {"id": "s016", "text": "तकनीक की मदद से आज हम घर बैठे दुनिया के किसी भी कोने से जुड़ सकते हैं।", "length_category": "medium"},
        # Long (25+ words)
        {"id": "s017", "text": "आर्टिफिशियल इंटेलिजेंस आज के दौर में बहुत तेज़ी से विकसित हो रहा है। इसका उपयोग स्वास्थ्य, शिक्षा, और व्यापार में किया जा रहा है।", "length_category": "long"},
        {"id": "s018", "text": "Hamara desh bahut vividhta se bhara hua hai. Yahan alag alag dharmon, bhashaon aur sanskritiyon ke log milke rehte hain aur ek dusre ka samman karte hain.", "length_category": "long"},
        {"id": "s019", "text": "पिछले कुछ वर्षों में डिजिटल भुगतान का चलन बहुत बढ़ा है। अब लोग नकद पैसों की जगह UPI और डेबिट कार्ड का इस्तेमाल करना ज्यादा पसंद करते हैं।", "length_category": "long"},
        {"id": "s020", "text": "Renewable energy, jaise solar aur wind power, ek behtar bhavishya ke liye bahut zaroori hai. Humein fossil fuels par apni nirbharta kam karni hogi aur safed urja ke srotron ko apnaana hoga.", "length_category": "long"},
        {"id": "s021", "text": "भारतीय सिनेमा ने पिछले सौ वर्षों में एक लंबा सफर तय किया है। मूक फिल्मों से लेकर आज की उच्च-तकनीक वाली फिल्मों तक, यह उद्योग लगातार बदलता और निखरता रहा है।", "length_category": "long"},
        {"id": "s022", "text": "Shiksha kisi bhi samaj ki buniyad hoti hai. Jab tak har bachche ko quality education nahi milti, tab tak hum ek sahi mayne mein viksit rashtra nahi ban sakte. Isliye sarkar aur nagarikon dono ko is disha mein milkar kaam karna hoga.", "length_category": "long"},
        {"id": "s023", "text": "जलवायु परिवर्तन आज विश्व की सबसे बड़ी चुनौतियों में से एक है। बढ़ते तापमान, अनियमित बारिश, और प्राकृतिक आपदाओं की बढ़ती संख्या इसके स्पष्ट संकेत हैं।", "length_category": "long"},
        {"id": "s024", "text": "Aaj ke digital zamane mein, social media ek bahut badi taakat ban gaya hai. Iska sahi istemal kiya jaye to yeh logon ko ek doosre se jodne aur sahi jaankari phailane mein bahut upyogi hai.", "length_category": "long"},
        {"id": "s025", "text": "स्वास्थ्य ही असली धन है, यह कहावत आज भी उतनी ही सच है। नियमित व्यायाम, संतुलित आहार, और पर्याप्त नींद — ये तीनों मिलकर एक स्वस्थ जीवन की नींव बनाते हैं।", "length_category": "long"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Report writer
# ─────────────────────────────────────────────────────────────────────────────

def save_tts_report(report: TTSPhaseReport, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{report.phase}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
    log.info(f"JSON report → {json_path}")

    md_path = output_dir / f"{report.phase}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_tts_markdown(report))
    log.info(f"Markdown report → {md_path}")


def _render_tts_markdown(report: TTSPhaseReport) -> str:
    phase_label = report.phase.upper()
    lines = [
        f"# TTS Benchmark — {phase_label}",
        f"",
        f"**Model:** `{report.model_name}`  ",
        f"**Reference audio:** `{report.reference_audio}`  ",
        f"**Timestamp:** {report.timestamp}",
        f"",
        f"## Summary",
        f"",
    ]
    for k, v in report.summary.items():
        label = k.replace("_", " ").capitalize()
        if isinstance(v, float):
            lines.append(f"- **{label}:** {v:.4f}")
        else:
            lines.append(f"- **{label}:** {v}")
    lines.append("")

    if report.phase == "tts_phase1":
        lines += _phase1_tts_table(report.results)
    elif report.phase == "tts_phase2":
        lines += _phase2_tts_table(report.results)

    return "\n".join(lines)


def _phase1_tts_table(results: list) -> list[str]:
    lines = [
        "## Per-sentence results",
        "",
        "| ID | Run | Latency (s) | Audio (s) | RTF | UTMOS | Silence | Clipping | IC-CER | W-CER |",
        "|----|-----|-------------|-----------|-----|-------|---------|----------|--------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['sentence_id']} | {r['run_index']}"
            f" | {r['latency_sec']:.3f}"
            f" | {r['generated_audio_sec']:.2f}"
            f" | {r['rtf']:.3f}"
            f" | {r['utmos_score']:.2f}"
            f" | {r['silence_ratio']:.3f}"
            f" | {r['clipping_ratio']:.4f}"
            f" | {r['indicconformer_cer']:.3f}"
            f" | {r['whisper_cer']:.3f} |"
        )
    return lines


def _phase2_tts_table(results: list) -> list[str]:
    lines = [
        "## Concurrency sweep results",
        "",
        "| Concurrency | Sentences | Total (s) | Mean lat (s) | P95 lat (s) | Sent/s | RTF | UTMOS | IC-CER | W-CER | Peak VRAM | GPU% |",
        "|-------------|-----------|-----------|--------------|-------------|--------|-----|-------|--------|-------|-----------|------|",
    ]
    for r in results:
        lines.append(
            f"| {r['concurrency_level']}"
            f" | {r['n_sentences']}"
            f" | {r['total_wall_time_sec']:.2f}"
            f" | {r['mean_latency_sec']:.3f}"
            f" | {r['p95_latency_sec']:.3f}"
            f" | {r['throughput_sentences_per_sec']:.2f}"
            f" | {r['mean_rtf']:.3f}"
            f" | {r['mean_utmos_score']:.2f}"
            f" | {r['mean_indicconformer_cer']:.3f}"
            f" | {r['mean_whisper_cer']:.3f}"
            f" | {r['gpu_peak_vram_mb']:.0f} MB"
            f" | {r['gpu_mean_util_pct']:.1f} |"
        )
    return lines