"""RunPod Serverless handler. Bundles voice/music separation (Demucs),
transcription (faster-whisper), NLLB translation, and OmniVoice voice
cloning into one auto-scaling endpoint (scales to zero when idle, spins up
on demand, billed per second of actual use — no manual pod management).

Input shape (event["input"]):
  {"mode": "separate", "audio_b64": "<base64 mp3>"}
  {"mode": "transcribe", "audio_b64": "<base64 mp3>"}
  {"mode": "translate", "texts": ["...", "..."], "target_lang_code": "eng_Latn"}
  {"mode": "generate", "text": "...", "ref_audio_b64": "<base64 wav>"}
"""
import base64
import tempfile
from pathlib import Path

import runpod
import soundfile as sf
import torch
from demucs.api import Separator, save_audio
from faster_whisper import WhisperModel
from omnivoice import OmniVoice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATE_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "arb_Arab"

_tts_model = None
_translate_model = None
_translate_tokenizer = None
_whisper_model = None
_demucs_separator = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return _whisper_model


def get_demucs_separator():
    global _demucs_separator
    if _demucs_separator is None:
        _demucs_separator = Separator(model="htdemucs", device="cuda")
    return _demucs_separator


def get_tts_model():
    global _tts_model
    if _tts_model is None:
        _tts_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32
        )
    return _tts_model


def get_translate_model():
    global _translate_model, _translate_tokenizer
    if _translate_model is None:
        _translate_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATE_MODEL_NAME, src_lang=SOURCE_LANG
        )
        _translate_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATE_MODEL_NAME, use_safetensors=True
        ).to("cuda:0")
    return _translate_model, _translate_tokenizer


def handle_translate(job_input):
    model, tokenizer = get_translate_model()
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(job_input["target_lang_code"])
    texts = job_input["texts"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
    generated = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=400)
    return {"translations": tokenizer.batch_decode(generated, skip_special_tokens=True)}


def handle_generate(job_input):
    model = get_tts_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
        ref_file.write(base64.b64decode(job_input["ref_audio_b64"]))
        ref_path = ref_file.name

    audio = model.generate(text=job_input["text"], ref_audio=ref_path)

    out_path = ref_path.replace(".wav", "_out.wav")
    sf.write(out_path, audio[0], 24000)
    audio_b64 = base64.b64encode(Path(out_path).read_bytes()).decode()

    Path(ref_path).unlink(missing_ok=True)
    Path(out_path).unlink(missing_ok=True)

    return {"audio_b64": audio_b64}


def handle_separate(job_input):
    separator = get_demucs_separator()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
        audio_file.write(base64.b64decode(job_input["audio_b64"]))
        audio_path = audio_file.name

    _, separated = separator.separate_audio_file(audio_path)
    vocals = separated["vocals"]
    background = sum(v for k, v in separated.items() if k != "vocals")

    vocals_path = audio_path.replace(".mp3", "_vocals.wav")
    background_path = audio_path.replace(".mp3", "_background.wav")
    save_audio(vocals, vocals_path, samplerate=separator.samplerate)
    save_audio(background, background_path, samplerate=separator.samplerate)

    result = {
        "vocals_b64": base64.b64encode(Path(vocals_path).read_bytes()).decode(),
        "background_b64": base64.b64encode(Path(background_path).read_bytes()).decode(),
    }

    for p in (audio_path, vocals_path, background_path):
        Path(p).unlink(missing_ok=True)

    return result


def handle_transcribe(job_input):
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
        audio_file.write(base64.b64decode(job_input["audio_b64"]))
        audio_path = audio_file.name

    segments_iter, _info = model.transcribe(audio_path, language="ar", word_timestamps=False)
    segments = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments_iter
    ]

    Path(audio_path).unlink(missing_ok=True)

    return {"segments": segments}


def handle_debug(job_input):
    import subprocess
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi_stdout": smi.stdout,
        "nvidia_smi_stderr": smi.stderr,
    }


def handler(event):
    job_input = event["input"]
    mode = job_input.get("mode")

    if mode == "separate":
        return handle_separate(job_input)
    elif mode == "transcribe":
        return handle_transcribe(job_input)
    elif mode == "translate":
        return handle_translate(job_input)
    elif mode == "generate":
        return handle_generate(job_input)
    elif mode == "debug":
        return handle_debug(job_input)
    else:
        return {
            "error": f"Unknown mode: {mode!r}. Use 'separate', 'transcribe', 'translate', 'generate', or 'debug'."
        }


runpod.serverless.start({"handler": handler})
