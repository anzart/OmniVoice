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
# A batched synthesize: returns (list of float32 waveforms, None) on success,
# or (None, error_message) on failure.
SynthesizeBatchFn = Callable[..., tuple[Optional[list], Optional[str]]]
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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Azul fell-awen.",
                    "language": "kab",
                    "mode": "tts",
                    "num_step": 32,
                    "guidance_scale": 2.0,
                }
            ]
        }
    }


class BatchTtsRequest(BaseModel):
    """Request body for POST /api/tts_batch — several texts, one shared voice.

    All settings/voice fields are shared across every text in ``texts``; only
    the text differs (e.g. the A/B comparison tool). The model runs them as a
    single batched forward pass, which is far cheaper than concurrent requests
    contending for one GPU/MPS device.
    """

    texts: list[str] = Field(..., description="Texts to synthesize (same voice).")
    language: Optional[str] = None
    instruct: Optional[str] = None
    mode: str = "tts"

    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    speed: Optional[float] = None
    duration: Optional[float] = None
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    position_temperature: float = 5.0
    class_temperature: float = 0.0

    ref_audio_base64: Optional[str] = None
    ref_text: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": ["idir", "iddir"],
                    "language": "kab",
                    "mode": "tts",
                    "num_step": 32,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error envelope returned on a generation or validation failure."""

    error: str = Field(..., description="Human-readable error message.")


class HealthResponse(BaseModel):
    """Server readiness + loaded-model info."""

    status: str = Field("ok", description='Always "ok" once the model is loaded.')
    model: str = Field(..., description="Loaded checkpoint id.")
    device: str = Field(..., description='Compute device: "cuda" | "mps" | "cpu".')
    samplingRate: int = Field(..., description="Native output sample rate (Hz).")


class TranscribeResponse(BaseModel):
    """ASR transcription result."""

    text: str = Field(..., description="Recognized transcript of the reference audio.")


class BatchTtsResponse(BaseModel):
    """Batched synthesis result — one WAV per input text, in the same order."""

    audios: list[str] = Field(
        ..., description="Base64-encoded 32-bit float WAVs, aligned with `texts`."
    )
    samplingRate: int = Field(..., description="Output sample rate (Hz) of each clip.")


# OpenAPI tag groups shown in the Swagger UI.
_TAGS_METADATA = [
    {"name": "Health", "description": "Server readiness and model info."},
    {"name": "TTS", "description": "Speech synthesis — single clip and batched."},
    {"name": "ASR", "description": "Reference-audio transcription (Whisper)."},
]


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
    synthesize_batch: Optional[SynthesizeBatchFn] = None,
) -> FastAPI:
    """Build the FastAPI app exposing the REST endpoints.

    The Gradio UI is mounted onto this same app by the caller (app.py), so
    `/api/*` and the Gradio interface at `/` live in one server / one port.
    """
    app = FastAPI(
        title="OmniVoice REST API",
        version="1.0.0",
        description=(
            "Local REST layer over the OmniVoice TTS model, consumed by the "
            "Dawal Studio frontend. Synthesize speech (single clip or batched "
            "for A/B comparison), design or clone voices, and transcribe "
            "reference audio. The Gradio demo UI is mounted at `/`."
        ),
        openapi_tags=_TAGS_METADATA,
    )

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

    @app.get(
        "/api/health",
        tags=["Health"],
        summary="Readiness + model info",
        response_model=HealthResponse,
    )
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

    @app.post(
        "/api/transcribe",
        tags=["ASR"],
        summary="Transcribe reference audio",
        response_model=TranscribeResponse,
        responses={422: {"model": ErrorResponse, "description": "Transcription failed"}},
    )
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

    @app.post(
        "/api/tts",
        tags=["TTS"],
        summary="Synthesize one clip",
        responses={
            200: {
                "content": {"audio/wav": {}},
                "description": "32-bit float WAV (audio/wav).",
            },
            422: {"model": ErrorResponse, "description": "Generation error"},
        },
    )
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

    @app.post(
        "/api/tts_batch",
        tags=["TTS"],
        summary="Synthesize several clips (batched, one voice)",
        response_model=BatchTtsResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Generation error"},
            501: {
                "model": ErrorResponse,
                "description": "Batch synthesis not available on this build",
            },
        },
    )
    async def tts_batch(req: BatchTtsRequest):
        """Synthesize several texts sharing one voice in a single batched pass.

        Returns ``{ "audios": [<base64 wav>, ...], "samplingRate": N }`` in the
        same order as ``texts``. 422 on a generation error; 501 when the server
        build doesn't provide a batched synthesize (frontend falls back).
        """
        if synthesize_batch is None:
            return JSONResponse(
                {"error": "Batch synthesis is not available on this server."},
                status_code=501,
            )
        tmp_path: Optional[str] = None
        try:
            if not req.texts:
                return JSONResponse({"error": "texts is required."}, status_code=422)

            ref_audio_path: Optional[str] = None
            if req.mode == "clone":
                if not req.ref_audio_base64:
                    return JSONResponse(
                        {"error": "ref_audio_base64 is required for clone mode."},
                        status_code=422,
                    )
                raw = base64.b64decode(req.ref_audio_base64)
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                ref_audio_path = tmp_path

            waveforms, error = synthesize_batch(
                texts=req.texts,
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

            audios_b64 = []
            for wf in waveforms:
                buf = io.BytesIO()
                sf.write(buf, np.asarray(wf, dtype=np.float32), sampling_rate,
                         format="WAV", subtype="FLOAT")
                audios_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

            return JSONResponse(
                {"audios": audios_b64, "samplingRate": sampling_rate},
                headers={"Cross-Origin-Resource-Policy": "cross-origin"},
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return app
