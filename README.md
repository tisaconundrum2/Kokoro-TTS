---
title: Kokoro TTS
emoji: ❤️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Upgraded to v1.0!
disable_embedding: true
---

Kokoro-TTS is a Gradio app for generating speech with Kokoro voices.

## Prerequisites

- Docker 24+ (recommended)
- Optional: NVIDIA Container Toolkit if you want GPU acceleration in Docker

## Build With Docker

From the repository root, build the image:

```bash
docker build -t kokoro-tts:latest .
```

## Run With Docker (CPU)

The app listens on port 7860 in the container.

```bash
docker run --rm -it -p 7860:7860 kokoro-tts:latest
```

Then open:

- http://localhost:7860

## Run With Docker (GPU, optional)

If your host supports NVIDIA containers:

```bash
docker run --rm -it --gpus all -p 7860:7860 kokoro-tts:latest
```

## Stop The Container

Press `Ctrl+C` in the terminal where the container is running.

## Notes

- The Docker image installs system dependencies used by the app, including `espeak-ng` and `ffmpeg`.
- During image build, spaCy model `en_core_web_sm` is downloaded.

Hugging Face Spaces config reference:
https://huggingface.co/docs/hub/spaces-config-reference