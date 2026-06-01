"""REST API layer for the OmniVoice server — local fork addition.

Wraps the already-loaded model in a small FastAPI app so the dawal-studio
frontend can call it over HTTP with a stable JSON contract, instead of going
through Gradio's queue/event protocol.

Two cross-origin concerns are handled here, because the frontend runs under
`Cross-Origin-Embedder-Policy: require-corp` (needed for SQLite WASM + the
AudioWorklet):
  * CORS — so the browser allows the cross-port request (5173 → 7860);
  * `Cross-Origin-Resource-Policy: cross-origin` on every response — so COEP
    doesn't block the JSON / WAV payload.

This file has no upstream counterpart; it lives only on the local-offline
branch and never conflicts on `git pull upstream`.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# A synthesize callable: returns (float32 waveform, None) on success, or
# (None, error_message) on failure. Provided by app.py so the API and the
# Gradio UI share one generation path.
SynthesizeFn = Callable[..., tuple[Optional[np.ndarray], Optional[str]]]
# Transcribe a WAV file path → text (the model's Whisper ASR).
TranscribeFn = Callable[[str], str]


class TranscribeRequest(BaseModel):
    """Request body for POST /api/transcribe."""

    audio_base64: str = Field(..., description="Base64-encoded WAV to transcribe.")


class TtsRequest(BaseModel):
    """Request body for POST /api/tts."""

    text: str = Field(..., description="Text to synthesize.")
    language: Optional[str] = Field(None, description="Language code, or null for auto.")
    instruct: Optional[str] = Field(None, description="Optional style instruction.")
    mode: str = Field("tts", description='"tts" (plain) or "clone" (voice cloning).')

    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    speed: Optional[float] = None
    duration: Optional[float] = None
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    # Sampling randomness — 0 = greedy/deterministic, higher = more varied.
    position_temperature: float = 5.0
    class_temperature: float = 0.0

    # Voice cloning (mode == "clone")
    ref_audio_base64: Optional[str] = Field(
        None, description="Base64-encoded reference WAV (required for clone mode)."
    )
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio.")


def _corp_cross_origin(response: Response) -> None:
    # COEP: require-corp on the frontend blocks cross-origin resources unless
    # they opt in with this header.
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"


def create_app(
    *,
    synthesize: SynthesizeFn,
    transcribe: TranscribeFn,
    sampling_rate: int,
    device: str,
    checkpoint: str,
) -> FastAPI:
    """Build the FastAPI app exposing the REST endpoints.

    The Gradio UI is mounted onto this same app by the caller (app.py), so
    `/api/*` and the Gradio interface at `/` live in one server / one port.
    """
    app = FastAPI(title="OmniVoice REST API", version="1.0.0")

    # CORS: this is a local dev tool with no auth/cookies, so a permissive
    # policy is fine. Override the allowed origins with OMNIVOICE_CORS_ORIGINS
    # (comma-separated) if you ever need to lock it down.
    origins_env = os.environ.get("OMNIVOICE_CORS_ORIGINS", "*")
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_corp_header(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        return response

    @app.get("/api/health")
    async def health():
        """Readiness probe — the model is loaded by the time this serves."""
        return JSONResponse(
            {
                "status": "ok",
                "model": checkpoint,
                "device": device,
                "samplingRate": sampling_rate,
            }
        )

    @app.post("/api/transcribe")
    async def transcribe_audio(req: TranscribeRequest):
        """Transcribe a reference WAV using the model's ASR → `{ "text": ... }`.

        Used by the frontend to pre-fill a clone voice's reference transcript.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(base64.b64decode(req.audio_base64))
            text = transcribe(tmp_path)
            return JSONResponse({"text": text})
        except Exception as e:
            return JSONResponse(
                {"error": f"{type(e).__name__}: {e}"}, status_code=422
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post("/api/tts")
    async def tts(req: TtsRequest):
        """Synthesize speech. Returns a 32-bit float WAV (audio/wav).

        On a generation error returns 422 with `{ "error": "..." }`.
        """
        ref_audio_path: Optional[str] = None
        tmp_path: Optional[str] = None
        try:
            if req.mode == "clone":
                if not req.ref_audio_base64:
                    return JSONResponse(
                        {"error": "ref_audio_base64 is required for clone mode."},
                        status_code=422,
                    )
                # create_voice_clone_prompt expects a file path; decode the
                # base64 WAV to a temp file and hand over its path.
                raw = base64.b64decode(req.ref_audio_base64)
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                ref_audio_path = tmp_path

            waveform, error = synthesize(
                text=req.text,
                language=req.language,
                ref_audio=ref_audio_path,
                instruct=req.instruct,
                num_step=req.num_step,
                guidance_scale=req.guidance_scale,
                denoise=req.denoise,
                speed=req.speed,
                duration=req.duration,
                preprocess_prompt=req.preprocess_prompt,
                postprocess_output=req.postprocess_output,
                position_temperature=req.position_temperature,
                class_temperature=req.class_temperature,
                mode=req.mode,
                ref_text=req.ref_text,
            )

            if error is not None:
                return JSONResponse({"error": error}, status_code=422)

            # Encode as 32-bit float WAV — lossless w.r.t. the model output
            # (the model emits float; int16 would throw away precision).
            buf = io.BytesIO()
            sf.write(buf, np.asarray(waveform, dtype=np.float32), sampling_rate,
                     format="WAV", subtype="FLOAT")
            data = buf.getvalue()

            return Response(
                content=data,
                media_type="audio/wav",
                headers={
                    "Cross-Origin-Resource-Policy": "cross-origin",
                    "X-Sampling-Rate": str(sampling_rate),
                    "Content-Disposition": 'inline; filename="tts.wav"',
                },
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app
