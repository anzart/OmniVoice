#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.

"""

import logging
import os
from typing import Any, Dict

# Force fully offline mode — models are cached locally, no internet needed
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

import numpy as np
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

print(f"Loading model from {CHECKPOINT} to {DEVICE} ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map=DEVICE,
    dtype=DTYPE,
    load_asr=True,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def _synthesize(
    text,
    language=None,
    ref_audio=None,
    instruct=None,
    num_step=32,
    guidance_scale=2.0,
    denoise=True,
    speed=None,
    duration=None,
    preprocess_prompt=True,
    postprocess_output=True,
    mode="tts",
    ref_text=None,
):
    """Core generation, shared by the Gradio UI and the REST API.

    Returns ``(float32_waveform, None)`` on success or ``(None, error)`` on
    failure. Float is kept (the model emits float) so callers decide whether
    to downsample to int16 — the REST API keeps float for fidelity.
    """
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    return np.asarray(audio[0], dtype=np.float32), None


def _gen_core(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
):
    # Gradio adapter: delegate to _synthesize, then pack the (rate, int16)
    # tuple + status string that gr.Audio expects.
    waveform, error = _synthesize(
        text=text,
        language=language,
        ref_audio=ref_audio,
        instruct=instruct,
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=denoise,
        speed=speed,
        duration=duration,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        mode=mode,
        ref_text=ref_text,
    )
    if error is not None:
        return None, error
    return (sampling_rate, (waveform * 32767).astype(np.int16)), "Done."


# ---------------------------------------------------------------------------
# Local wrapper (no ZeroGPU needed)
# ---------------------------------------------------------------------------


def generate_fn(*args, **kwargs):
    return _gen_core(*args, **kwargs)


# ---------------------------------------------------------------------------
# Build and launch demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)

if __name__ == "__main__":
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

    if os.environ.get("OMNIVOICE_API", "1") == "1":
        # Serve the REST API and the Gradio UI from one FastAPI app / one port:
        #   /api/*  → REST endpoints (consumed by the dawal-studio frontend)
        #   /       → the Gradio demo UI
        # API routes are registered before mounting Gradio, so they win on match.
        import uvicorn
        import gradio as gr
        from api import create_app

        fastapi_app = create_app(
            synthesize=_synthesize,
            sampling_rate=sampling_rate,
            device=DEVICE,
            checkpoint=CHECKPOINT,
        )
        fastapi_app = gr.mount_gradio_app(fastapi_app, demo.queue(), path="/")
        print(f"REST API on http://{host}:{port}/api  |  Gradio UI on http://{host}:{port}/")
        uvicorn.run(fastapi_app, host=host, port=port)
    else:
        demo.queue().launch(server_name=host, server_port=port)
