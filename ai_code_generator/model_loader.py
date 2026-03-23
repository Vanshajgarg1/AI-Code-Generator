"""
model_loader.py — HuggingFace Model Loading
============================================
Responsible for:
  - Reading configuration from environment variables
  - Downloading / caching the model and tokenizer from HuggingFace Hub
  - Providing a thin singleton wrapper so the heavy model is only loaded once
  - Supporting both LOCAL inference and the free HuggingFace Inference API
"""

import os
import logging
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration from .env ───────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "Salesforce/codegen-350M-mono")
HF_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_API_TOKEN") or None
FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
INFERENCE_MODE: str = os.getenv("INFERENCE_MODE", "local").lower()

# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> str:
    """
    Detect the best available compute device.

    Priority: CUDA GPU → Apple MPS → CPU
    The FORCE_CPU env var overrides this to always return 'cpu'.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if FORCE_CPU:
        logger.info("FORCE_CPU=true → using CPU.")
        return "cpu"

    if torch.cuda.is_available():
        logger.info("CUDA GPU detected → using GPU.")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS detected → using MPS.")
        return "mps"

    logger.info("No GPU found → falling back to CPU.")
    return "cpu"


# ── Singleton model cache ─────────────────────────────────────────────────────
_cached_pipeline = None  # Holds the loaded pipeline so it's not reloaded


def load_model() -> "pipeline":
    """
    Load the code-generation pipeline exactly once and cache it.

    Uses the HuggingFace `pipeline` abstraction which bundles:
      - The tokenizer (converts text → token IDs)
      - The model (predicts next tokens)
      - The decoding logic (converts token IDs → text)

    The first call downloads the model weights (~350 MB for the default model).
    Subsequent calls return the cached pipeline immediately.

    Returns:
        A HuggingFace text-generation pipeline ready for inference.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _cached_pipeline

    # Return the already-loaded pipeline if available
    if _cached_pipeline is not None:
        logger.info("Returning cached model pipeline.")
        return _cached_pipeline

    device = get_device()

    logger.info("Loading model: %s  |  device: %s", MODEL_NAME, device)
    logger.info("This may take a few minutes on the first run (downloading weights)…")

    try:
        # ── Step 1: Load the tokenizer ────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,        # Required for gated/private models
            trust_remote_code=True, # Needed for some custom architectures
        )

        # Some tokenizers don't have a pad token by default; use EOS as pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Step 2: Load the model weights ────────────────────────────────────
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
            low_cpu_mem_usage=True,    # Reduces peak RAM during loading
        )

        # Move model to the target device
        model = model.to(device)
        model.eval()  # Disable dropout for deterministic inference

        # ── Step 3: Wrap into a text-generation pipeline ─────────────────────
        _cached_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,  # -1 → CPU for pipeline
        )

        logger.info("✅ Model loaded successfully.")
        return _cached_pipeline

    except Exception as exc:
        logger.error("❌ Failed to load model '%s': %s", MODEL_NAME, exc)
        raise RuntimeError(
            f"Could not load model '{MODEL_NAME}'.\n"
            "Possible fixes:\n"
            "  1. Check your internet connection (first-time download).\n"
            "  2. Set HUGGINGFACE_API_TOKEN in .env for gated models.\n"
            "  3. Try a smaller model like 'Salesforce/codegen-350M-mono'.\n"
            f"Original error: {exc}"
        ) from exc


def get_model_info() -> dict:
    """
    Return metadata about the currently configured model.

    Returns:
        Dictionary with model name, device, and inference mode.
    """
    return {
        "model_name": MODEL_NAME,
        "device": get_device(),
        "inference_mode": INFERENCE_MODE,
        "hf_token_set": HF_TOKEN is not None,
    }
