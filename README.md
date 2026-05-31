# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

OmniVoice is a state-of-the-art massively multilingual zero-shot text-to-speech (TTS) model supporting over 600 languages. Built on a novel diffusion language model-style architecture, it generates high-quality speech with superior inference speed, supporting voice cloning and voice design.

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Training & Evaluation](#training--evaluation) | [Discussion](#discussion--communication) | [Citation](#citation)

## Key Features

- **600+ Languages Supported**: The broadest language coverage among zero-shot TTS models ([full list](docs/languages.md)).
- **Voice Cloning**: State-of-the-art voice cloning quality.
- **Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.).
- **Fine-grained Control**: Non-verbal symbols (e.g., `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (40x faster than real-time).
- **Diffusion Language Model-style Architecture**: A clean, streamlined, and scalable design that delivers both quality and speed.

---

## Installation

Choose **one** of the following methods: **pip** or **uv**.

### pip

> We recommend using a fresh virtual environment (e.g., `conda`, `venv`, etc.) to avoid conflicts.

**Step 1**: Install PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# Install pytorch with your CUDA version, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
> See [PyTorch official site](https://pytorch.org/get-started/locally/) for other versions installation.

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

</details>

<details>
<summary>Intel Arc GPU (XPU)</summary>

Intel Arc GPUs (Alchemist and Battlemage architectures) are supported via PyTorch's XPU backend.

1. Install the [Intel GPU drivers](https://dgpu-docs.intel.com/driver/installation.html) for your OS.

2. Install PyTorch with XPU support from Intel's wheel index:

```bash
pip install torch torchaudio --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

> See [Intel's PyTorch XPU guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) for version-specific instructions.

3. Verify the backend is working:

```bash
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
```

**Notes**:
- `flash_attn` is not available on XPU; the model automatically falls back to SDPA.
- Training with packed sequences (`flex_attention`) has partial XPU support; single-GPU SDPA training should work.
- Tested on Arc A310 (Alchemist, 4 GB) and Arc Pro B50 (Battlemage, 16 GB).

</details>

**Step 2**: Install OmniVoice (choose one)

```bash
# From PyPI (stable release)
pip install omnivoice

# From the latest source on GitHub (no need to clone)
pip install git+https://github.com/k2-fsa/OmniVoice.git

# For development (clone first, editable install)
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### uv

Clone the repository and sync dependencies:

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync
```

> **Tip**: Can use mirror with `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

---

## Quick Start

Try OmniVoice without coding:

- Launch the local web UI: `omnivoice-demo --ip 0.0.0.0 --port 8001`

- Or try it directly on [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)

- Or run it in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb)

> If you have trouble connecting to HuggingFace when downloading the pre-trained models, set `export HF_ENDPOINT="https://hf-mirror.com"` before running.

For full usage, see the [Python API](#python-api) and [Command-Line Tools](#command-line-tools) sections below.

---

## Python API

OmniVoice supports three generation modes. All features in this section are also available via [command-line tools](#command-line-tools).

### Voice Cloning

Clone a voice from a short reference audio. Provide `ref_audio` and `ref_text`:

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead
# Intel Arc GPU users: use device_map="xpu" instead

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

sf.write("out.wav", audio[0], 24000)
```

> **Tips**
>
> - Use a 3–10 seconds reference audio clip. Longer audio slows down inference and may degrade cloning quality.
> - For standard pronunciation, use a reference audio in the **same language** as the target speech. In cross-lingual voice cloning (i.e., the reference audio and target speech are in different languages), the generated speech will carry an accent from the reference audio's language.
> - For better results with Arabic numerals, normalize them to words first (e.g., "123" → "one hundred twenty-three") with text normalization tools (e.g., [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)).
>
> For more tips, see [docs/tips.md](docs/tips.md).

### Voice Design

Describe the desired voice with speaker attributes — no reference audio needed.
Supported attributes: **gender** (male/female), **age** (child to elderly),
**pitch** (very low to very high), **style** (whisper), **English accent**
(American, British, etc.), and **Chinese dialect** (四川话, 陕西话, etc.).
Attributes are comma-separated and freely combinable across categories.

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

> **Note**: The model is primarily trained on the voice cloning task, so voice cloning is the most stable mode. Voice design is trained on Chinese and English data only. It can generalize to other languages, but may produce unstable results for some low-resource languages or edge cases.

See [docs/voice-design.md](docs/voice-design.md) for the full attribute
reference, Chinese equivalents, and usage tips.

### Auto Voice

Let the model choose a voice automatically:

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

### Generation Parameters

All above three modes share the same `model.generate()` API. You can further control the generation behavior via keyword arguments:

```python
audio = model.generate(
    text="...",
    num_step=32,  # diffusion steps (or 16 for faster inference)
    speed=1.0,     # speed factor (>1.0 faster, <1.0 slower)
    duration=10.0, # fixed output duration in seconds (overrides speed)
    # ... more options
)
```
See more detailed control in [docs/generation-parameters.md](docs/generation-parameters.md).

### Non-Verbal & Pronunciation Control

OmniVoice supports inline **non-verbal symbols** and **pronunciation correction** within the input text.

**Non-verbal symbols**: Insert tags like `[laughter]` directly in the text to add expressive non-verbal sounds.

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

Supported tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.

**Pronunciation control (Chinese)**: Use pinyin with tone numbers to correct specific character pronunciations.

```python
audio = model.generate(text="这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。")
```

**Pronunciation control (English)**: Use [CMU pronunciation dictionary](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)  (uppercase, in brackets) to override default English pronunciations.

```python
audio = model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")
```

---

## Command-Line Tools

Three CLI entry points are provided. The CLI tools support all features available in the Python API (voice cloning, voice design, auto voice, generation parameters, etc.) — all controlled via command-line arguments.

| Command | Description | Source |
|---|---|---|
| `omnivoice-demo` | Interactive Gradio web demo | [omnivoice/cli/demo.py](omnivoice/cli/demo.py) |
| `omnivoice-infer` | Single-item inference | [omnivoice/cli/infer.py](omnivoice/cli/infer.py) |
| `omnivoice-infer-batch` | Batch inference across multiple GPUs | [omnivoice/cli/infer_batch.py](omnivoice/cli/infer_batch.py) |

### Demo

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

Provides a web UI for voice cloning and voice design. See `omnivoice-demo --help` for all options.

### Single Inference

```bash
# Voice Cloning
# ref_text can be omitted (Whisper will auto-transcribe ref_audio to get it).
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# Voice Design
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# Auto Voice
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech."\
    --output hello.wav
```

### Batch Inference

`omnivoice-infer-batch` can distribute batch inference across multiple GPUs, designed for large-scale TTS tasks.

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

The test list is a JSONL file where each line is a JSON object:
```json
{"id": "sample_001", "text": "Hello world", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript", "instruct": "female, british accent", "language_id": "en", "duration": 10.0, "speed": 1.0}
```
Only `id` and `text` are mandatory fields. `ref_audio` and `ref_text` are used in voice cloning mode. `instruct` is used in voice design mode. If no reference audio or instruct are provided, the model will generate text in a random voice.

`language_id`, `duration`, and `speed` are optional. `duration` (in seconds) fixes the output length; `speed` controls the speaking rate. If `duration` and `speed` are both provided, `speed` will be ignored.

---

## Training & Evaluation

See [examples/](examples/) for the complete pipeline — from data preparation to training, evaluation, and finetuning.

---

## Discussion & Communication

You can directly discuss on [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

---

## Community Projects

OmniVoice is supported by a growing ecosystem of community projects.
Explore them in [Community Projects](docs/community-projects.md).

---

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

---

## Disclaimer

Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams, or any other illegal or unethical activities. All users shall ensure full compliance with applicable local laws, regulations, and ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications.

---

<!-- ========================================================================
     LOCAL FORK NOTES — not part of upstream k2-fsa/OmniVoice.
     Lives only on the `local-offline` branch. Kept at the very end of the
     file so `git rebase upstream/master` rarely touches it.
     ======================================================================== -->

## 🛠️ Local fork (dawal-studio integration)

This copy lives inside the **dawal-studio** project as a standalone Gradio TTS
server. It runs **fully offline** at runtime (the model is loaded from the
local Hugging Face cache; `app.py` sets `HF_HUB_OFFLINE=1`).

### What differs from upstream

Three **additive** files exist only here (committed on the `local-offline`
branch) — upstream has none of them, so they never conflict on a pull:

| File | Purpose |
|---|---|
| `app.py` | Gradio entry point. Forces offline mode, auto-detects **MPS/CPU** (falls back from CUDA), drops the ZeroGPU `spaces` decorator. |
| `requirements.txt` | Pip dependency list used to build the local venv (upstream uses `pyproject.toml` + `uv.lock`). |
| `run.sh` | Dev launcher: cd's into this folder, checks the venv, runs `app.py` with `GRADIO_SERVER_PORT=7860`. |

### First-time setup (recreate the virtualenv)

The `env/` virtualenv is **git-ignored** and must be created locally:

```bash
cd omnivoice-server
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv env
env/bin/python -m pip install -r requirements.txt
```

The model weights (`k2-fsa/OmniVoice`) must already be in the Hugging Face
cache (`~/.cache/huggingface`). With offline mode on, they are never
downloaded at runtime.

### Running the server

```bash
# Standalone, from the dawal-studio root:
npm run dev:server          # → http://127.0.0.1:7860

# Or directly:
bash omnivoice-server/run.sh
```

It is also started **automatically** by the dawal-studio dev command, in
parallel with the Vite web app:

```bash
npm run dev                 # WEB (Vite :5173) + TTS (Gradio :7860)
npm run dev:web             # web app only (skip the ~30 s model load)
```

`npm run dev` uses `concurrently`, so **Ctrl-C stops both** processes.

### REST API

On top of the Gradio UI, the local fork exposes a small **REST API** (see
`api.py`) so the dawal-studio frontend can call it with a stable JSON contract
instead of Gradio's queue/event protocol. Both share one port:

| Path | What |
|---|---|
| `GET /api/health` | `{ status, model, device, samplingRate }` — readiness probe. |
| `POST /api/tts` | Synthesize speech → `audio/wav` (32-bit float, mono). |
| `/` | The Gradio demo UI (unchanged). |

`POST /api/tts` body (JSON):

```jsonc
{
  "text": "Azul fell-awen",   // required
  "language": null,            // null = auto-detect
  "mode": "tts",               // "tts" or "clone"
  "instruct": null,
  "num_step": 32,
  "guidance_scale": 2.0,
  "denoise": true,
  "speed": null,
  "duration": null,
  // voice cloning (mode == "clone"):
  "ref_audio_base64": null,    // base64 WAV of the reference voice
  "ref_text": null
}
```

Errors come back as `{ "error": "..." }` with HTTP 422.

The responses carry CORS + `Cross-Origin-Resource-Policy: cross-origin`, which
is **required**: the frontend runs under `COEP: require-corp`, which would
otherwise block this cross-origin (`:5173` → `:7860`) response.

Environment toggles:

- `OMNIVOICE_API=0` — serve only the Gradio UI (no REST layer).
- `OMNIVOICE_CORS_ORIGINS` — comma-separated allowed origins (default `*`).

### Pulling updates from upstream (k2-fsa/OmniVoice)

Remotes:

- `upstream` → `https://github.com/k2-fsa/OmniVoice.git` (source of updates)
- `origin`   → `https://github.com/anzart/OmniVoice.git` (personal fork, patch backup)

To bring in remote changes (needs internet **only during the fetch**; runtime
stays offline) while keeping the local offline patch on top:

```bash
cd omnivoice-server
git fetch upstream
git rebase upstream/master        # replays the local-offline commit on top
# if a conflict appears (rare — only on the omnivoice/ package, docs/, examples/):
#   resolve the files, then:  git rebase --continue

# if requirements changed upstream, refresh the venv:
env/bin/python -m pip install -r requirements.txt
```

Optionally back up the patch online: `git push origin local-offline`.

> **Note:** the parent **dawal-studio** repo git-ignores this entire folder —
> `omnivoice-server/` is managed as its own independent git repository.
