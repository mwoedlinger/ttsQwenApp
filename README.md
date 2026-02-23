# Qwen3 TTS Studio

A minimal browser UI for zero-shot voice cloning with [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).

Register a voice from a short reference clip, then synthesize arbitrary text in that voice.

## Features

- **Voice registration** — record directly in the browser or upload an audio file (WAV, MP3, FLAC, OGG, M4A)
- **Voice cloning** — generates speech in the registered voice using Qwen3-TTS-12Hz-1.7B-Base
- **Multi-language** — auto-detects or lets you pin English, Chinese, German, Japanese, and more
- **Persistent profiles** — voice prompts are cached in memory and metadata is stored in `voice_profiles.json`
- Runs on CUDA, Apple Silicon (MPS), or CPU

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- The [`qwen-tts`](https://github.com/QwenLM/Qwen3-TTS) package

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Flash Attention 2** (optional, CUDA only): `pip install flash-attn --no-build-isolation`

## Run

```bash
python app.py
```

Then open [http://localhost:8000](http://localhost:8000).

The model (~3 GB) downloads from Hugging Face on first launch. A status pill in the header shows when it's ready.

## Usage

1. **Register a voice** (left panel)
   - Record 3–30 seconds of speech, or upload a clip
   - Provide the exact transcript of what is spoken
   - Click *Register Voice* (takes a few seconds while the model builds the voice prompt)

2. **Generate speech** (right panel)
   - Select a registered voice
   - Type or upload the text to synthesize
   - Click *Generate Speech*, then play or download the result

## Optional: `torch.compile`

Set `QWEN_TTS_COMPILE=1` to compile the autoregressive LM with `torch.compile`. This reduces per-token latency after a one-time warmup cost, and is most effective on CUDA.

```bash
QWEN_TTS_COMPILE=1 python app.py
```

## Project layout

```
app.py               FastAPI backend
static/index.html    Single-page frontend (no build step)
requirements.txt
uploads/             Reference audio files (created at runtime)
outputs/             Generated WAV files  (created at runtime)
voice_profiles.json  Voice metadata       (created at runtime)
```
