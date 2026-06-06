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
from omnivoice.utils.common import fix_random_seed

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


def _make_gen_config(
    num_step,
    guidance_scale,
    denoise,
    preprocess_prompt,
    postprocess_output,
    position_temperature,
    class_temperature,
    t_shift=None,
    layer_penalty_factor=None,
    audio_chunk_duration=None,
    audio_chunk_threshold=None,
):
    """Build an OmniVoiceGenerationConfig from raw (possibly None) values."""
    return OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
        position_temperature=float(position_temperature)
        if position_temperature is not None
        else 5.0,
        class_temperature=float(class_temperature)
        if class_temperature is not None
        else 0.0,
        t_shift=float(t_shift) if t_shift is not None else 0.1,
        layer_penalty_factor=float(layer_penalty_factor)
        if layer_penalty_factor is not None
        else 5.0,
        audio_chunk_duration=float(audio_chunk_duration)
        if audio_chunk_duration is not None
        else 15.0,
        audio_chunk_threshold=float(audio_chunk_threshold)
        if audio_chunk_threshold is not None
        else 30.0,
    )


def _build_generate_kwargs(
    *,
    text,
    language,
    ref_audio,
    instruct,
    speed,
    duration,
    mode,
    ref_text,
    gen_config,
):
    """Assemble the kwargs for ``model.generate`` shared by single + batch.

    ``text`` may be a string (single) or a list (batch). The voice prompt /
    instruct are built once and broadcast across the batch by the model. Raises
    ValueError with a user-facing message on bad input.
    """
    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text, language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            raise ValueError("Please upload a reference audio.")
        # One prompt — the model broadcasts it to every text in the batch.
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=bool(gen_config.preprocess_prompt),
        )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    return kw


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
    position_temperature=5.0,
    class_temperature=0.0,
    mode="tts",
    ref_text=None,
    t_shift=None,
    layer_penalty_factor=None,
    audio_chunk_duration=None,
    audio_chunk_threshold=None,
    seed=None,
):
    """Core generation, shared by the Gradio UI and the REST API.

    Returns ``(float32_waveform, None)`` on success or ``(None, error)`` on
    failure. Float is kept (the model emits float) so callers decide whether
    to downsample to int16 — the REST API keeps float for fidelity.

    ``seed`` (when > 0) seeds the global RNG so identical inputs reproduce the
    same audio; 0 / None leaves sampling random.
    """
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    if seed is not None and int(seed) > 0:
        fix_random_seed(int(seed))

    gen_config = _make_gen_config(
        num_step, guidance_scale, denoise, preprocess_prompt,
        postprocess_output, position_temperature, class_temperature,
        t_shift=t_shift, layer_penalty_factor=layer_penalty_factor,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )
    try:
        kw = _build_generate_kwargs(
            text=text.strip(), language=language, ref_audio=ref_audio,
            instruct=instruct, speed=speed, duration=duration, mode=mode,
            ref_text=ref_text, gen_config=gen_config,
        )
        audio = model.generate(**kw)
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    return np.asarray(audio[0], dtype=np.float32), None


def _synthesize_batch(
    texts,
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
    position_temperature=5.0,
    class_temperature=0.0,
    mode="tts",
    ref_text=None,
    t_shift=None,
    layer_penalty_factor=None,
    audio_chunk_duration=None,
    audio_chunk_threshold=None,
    seed=None,
):
    """Batched generation: ONE ``model.generate`` call for several texts that
    share the same voice + settings (e.g. the A/B comparison tool).

    A batched forward pass amortises the per-step cost instead of running N
    separate inferences that would contend for a single GPU/MPS device.

    Returns ``(list[float32_waveform], None)`` — one per input text, same order
    — or ``(None, error)``.
    """
    if not texts:
        return None, "Please enter the text to synthesize."
    cleaned = [(t or "").strip() for t in texts]
    if not all(cleaned):
        return None, "Please enter the text to synthesize."

    if seed is not None and int(seed) > 0:
        fix_random_seed(int(seed))

    gen_config = _make_gen_config(
        num_step, guidance_scale, denoise, preprocess_prompt,
        postprocess_output, position_temperature, class_temperature,
        t_shift=t_shift, layer_penalty_factor=layer_penalty_factor,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )
    try:
        kw = _build_generate_kwargs(
            text=cleaned, language=language, ref_audio=ref_audio,
            instruct=instruct, speed=speed, duration=duration, mode=mode,
            ref_text=ref_text, gen_config=gen_config,
        )
        audios = model.generate(**kw)
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    return [np.asarray(a, dtype=np.float32) for a in audios], None


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
            synthesize_batch=_synthesize_batch,
            transcribe=model.transcribe,
            sampling_rate=sampling_rate,
            device=DEVICE,
            checkpoint=CHECKPOINT,
        )
        fastapi_app = gr.mount_gradio_app(fastapi_app, demo.queue(), path="/")
        print(f"REST API on http://{host}:{port}/api  |  Gradio UI on http://{host}:{port}/")
        uvicorn.run(fastapi_app, host=host, port=port)
    else:
        demo.queue().launch(server_name=host, server_port=port)
