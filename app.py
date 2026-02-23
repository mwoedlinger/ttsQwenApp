"""
Qwen3-TTS Studio — FastAPI backend.

Workflow:
  1. Model (Qwen3-TTS-12Hz-1.7B-Base) loads at startup in a background thread.
  2. Users register voice profiles by uploading a reference audio clip + transcript.
     The model builds a reusable `voice_clone_prompt` and caches it in memory.
  3. Users call /api/generate with a voice name + text to synthesize new audio.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Directories / persistence
# ---------------------------------------------------------------------------
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
PROFILES_FILE = Path("voice_profiles.json")

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Loaded from disk on startup; updated on every register/delete.
_profiles_meta: dict[str, dict] = (
    json.loads(PROFILES_FILE.read_text()) if PROFILES_FILE.exists() else {}
)


def _save_profiles() -> None:
    PROFILES_FILE.write_text(json.dumps(_profiles_meta, indent=2))


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
_model = None
_model_status: str = "not_loaded"   # not_loaded | loading | ready | error
_model_error: Optional[str] = None
_model_lock = threading.Lock()

# In-memory voice-prompt cache.  Rebuilt from disk if process restarts.
_voice_prompts: dict[str, object] = {}


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_bg() -> None:
    global _model, _model_status, _model_error
    _model_status = "loading"
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore

        device = _get_device()
        # bf16 works on MPS (Apple Silicon) with PyTorch ≥ 2.x and cuts memory
        # bandwidth by 2x vs float32 — the dominant cost for autoregressive LMs.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        kwargs: dict = {"device_map": device, "dtype": dtype}
        if "cuda" in device:
            kwargs["attn_implementation"] = "flash_attention_2"
        elif device == "mps":
            # The model has _supports_sdpa=True; MPS exposes Metal-optimised SDPA.
            kwargs["attn_implementation"] = "sdpa"

        _model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            **kwargs,
        )

        # Optional: torch.compile the talker LM for faster autoregressive steps.
        # Set QWEN_TTS_COMPILE=1 to enable; disabled by default because MPS
        # compile support is still evolving and adds a cold-start cost.
        if os.environ.get("QWEN_TTS_COMPILE") == "1":
            try:
                backend = "aot_eager" if device == "mps" else "inductor"
                _model.model.talker = torch.compile(
                    _model.model.talker, backend=backend, dynamic=True
                )
            except Exception as compile_exc:
                print(f"[warn] torch.compile skipped: {compile_exc}")

        _model_status = "ready"
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        _model_status = "error"


def _get_or_build_prompt(name: str) -> object:
    """Return cached prompt or rebuild it from the stored audio/transcript."""
    if name in _voice_prompts:
        return _voice_prompts[name]

    meta = _profiles_meta.get(name)
    if meta is None:
        raise HTTPException(404, f"Voice '{name}' not found")

    audio_path = UPLOADS_DIR / meta["audio_file"]
    if not audio_path.exists():
        raise HTTPException(500, f"Reference audio for '{name}' is missing on disk")

    with _model_lock:
        # Double-check after acquiring lock
        if name in _voice_prompts:
            return _voice_prompts[name]
        prompt = _model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=meta["transcript"],
        )
        _voice_prompts[name] = prompt
    return prompt


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS Studio")


@app.on_event("startup")
async def _startup() -> None:
    t = threading.Thread(target=_load_model_bg, daemon=True)
    t.start()


# ---- Status ----------------------------------------------------------------

@app.get("/api/status")
def status() -> dict:
    return {
        "model_status": _model_status,
        "model_error": _model_error,
        "device": _get_device(),
        "voices": list(_profiles_meta.keys()),
    }


# ---- Voice profiles --------------------------------------------------------

@app.get("/api/voices")
def list_voices() -> dict:
    return {
        name: {
            "transcript_preview": meta["transcript"][:120],
            "created_at": meta["created_at"],
        }
        for name, meta in _profiles_meta.items()
    }


@app.post("/api/voices", status_code=201)
async def register_voice(
    name: str = Form(...),
    transcript: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    if _model_status != "ready":
        raise HTTPException(503, f"Model not ready yet ({_model_status}). Please wait.")

    name = name.strip()
    transcript = transcript.strip()

    if not name:
        raise HTTPException(400, "Voice name must not be empty")
    if not transcript:
        raise HTTPException(400, "Transcript must not be empty")
    if name in _profiles_meta:
        raise HTTPException(409, f"Voice '{name}' already exists. Delete it first.")

    # Persist uploaded audio
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    audio_filename = f"{uuid.uuid4()}{suffix}"
    audio_path = UPLOADS_DIR / audio_filename
    audio_path.write_bytes(await audio.read())

    # Build voice-clone prompt (runs model inference — can take a few seconds)
    try:
        with _model_lock:
            prompt = _model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=transcript,
            )
    except Exception as exc:  # noqa: BLE001
        audio_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to build voice prompt: {exc}") from exc

    _voice_prompts[name] = prompt
    _profiles_meta[name] = {
        "audio_file": audio_filename,
        "transcript": transcript,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_profiles()
    return {"status": "ok", "name": name}


@app.delete("/api/voices/{name}")
def delete_voice(name: str) -> dict:
    if name not in _profiles_meta:
        raise HTTPException(404, f"Voice '{name}' not found")

    meta = _profiles_meta.pop(name)
    _voice_prompts.pop(name, None)
    (UPLOADS_DIR / meta["audio_file"]).unlink(missing_ok=True)
    _save_profiles()
    return {"status": "ok"}


# ---- Generation ------------------------------------------------------------

class GenerateRequest(BaseModel):
    voice: str
    text: str
    language: str = "Auto"


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    if _model_status != "ready":
        raise HTTPException(503, f"Model not ready yet ({_model_status}). Please wait.")

    req.text = req.text.strip()
    if not req.text:
        raise HTTPException(400, "Text must not be empty")

    prompt = _get_or_build_prompt(req.voice)

    try:
        with _model_lock:
            wavs, sr = _model.generate_voice_clone(
                text=req.text,
                language=req.language,
                voice_clone_prompt=prompt,
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Generation failed: {exc}") from exc

    out_name = f"{uuid.uuid4()}.wav"
    out_path = OUTPUTS_DIR / out_name
    sf.write(str(out_path), wavs[0], sr)

    return {"audio_url": f"/api/audio/{out_name}", "filename": out_name}


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(path), media_type="audio/wav")


# ---- Static frontend -------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
