"""
code_generator.py — Core Code Generation Logic
===============================================
This module is the heart of the AI Code Generator.

Flow:
  1. Accept a plain-English user instruction.
  2. Build a structured prompt using utils.build_prompt().
  3. Send the prompt to the loaded HuggingFace model pipeline.
  4. Post-process the raw output into clean, formatted code.
  5. Optionally auto-format with Black.

Two inference backends are supported:
  - LOCAL  : Runs the model on your machine (CPU/GPU).
  - API    : Calls the free HuggingFace Inference API (no GPU needed).
"""

import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

from model_loader import load_model, MODEL_NAME, INFERENCE_MODE, HF_TOKEN
from utils import (
    build_prompt,
    extract_code_block,
    format_with_black,
    validate_prompt,
    get_language_from_prompt,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ── Generation hyper-parameters (read from .env with sensible defaults) ───────
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "300"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P: float = float(os.getenv("TOP_P", "0.95"))
REPETITION_PENALTY: float = float(os.getenv("REPETITION_PENALTY", "1.1"))

# HuggingFace Inference API endpoint template
HF_API_URL = "https://api-inference.huggingface.co/models/{model}"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_code(
    user_instruction: str,
    language: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    auto_format: bool = True,
) -> dict:
    """
    Generate code from a natural-language instruction.

    Args:
        user_instruction:  Plain-English programming task.
        language:          Target language (auto-detected if None).
        max_new_tokens:    Maximum number of tokens to generate.
        temperature:       Sampling temperature (lower = more deterministic).
        top_p:             Nucleus sampling threshold.
        repetition_penalty: Penalty for repeated tokens.
        auto_format:       Whether to run Black on the output (Python only).

    Returns:
        A dictionary containing:
          - "code"      : The generated source code string.
          - "language"  : Detected / specified language.
          - "prompt"    : The prompt that was sent to the model.
          - "error"     : None if successful, or an error message string.
          - "backend"   : "local" or "api".
    """
    # ── 1. Validate the user input ─────────────────────────────────────────
    is_valid, error_msg = validate_prompt(user_instruction)
    if not is_valid:
        return {"code": "", "language": "", "prompt": "", "error": error_msg, "backend": INFERENCE_MODE}

    # ── 2. Detect language if not specified ───────────────────────────────
    if language is None:
        language = get_language_from_prompt(user_instruction)
    logger.info("Detected language: %s", language)

    # ── 3. Build the prompt ────────────────────────────────────────────────
    prompt = build_prompt(user_instruction, language)
    logger.info("Prompt built (%d chars).", len(prompt))

    # ── 4. Generate code via selected backend ─────────────────────────────
    try:
        if INFERENCE_MODE == "api":
            raw_output = _generate_via_api(prompt, max_new_tokens, temperature, top_p)
        else:
            raw_output = _generate_locally(
                prompt, max_new_tokens, temperature, top_p, repetition_penalty
            )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return {
            "code": "",
            "language": language,
            "prompt": prompt,
            "error": str(exc),
            "backend": INFERENCE_MODE,
        }

    # ── 5. Post-process the raw output ────────────────────────────────────
    code = extract_code_block(raw_output, prompt)

    # ── 6. Auto-format Python output with Black ───────────────────────────
    if auto_format and language == "python" and code:
        code = format_with_black(code)

    logger.info("Code generated successfully (%d chars).", len(code))

    return {
        "code": code,
        "language": language,
        "prompt": prompt,
        "error": None,
        "backend": INFERENCE_MODE,
    }


# ── Local inference ───────────────────────────────────────────────────────────

def _generate_locally(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """
    Run the model locally using the cached HuggingFace pipeline.

    Args:
        prompt:            The formatted prompt string.
        max_new_tokens:    Token generation budget.
        temperature:       Sampling temperature.
        top_p:             Nucleus sampling p value.
        repetition_penalty: Token repetition penalty.

    Returns:
        Raw generated text from the model (includes the prompt echo).
    """
    logger.info("Running LOCAL inference …")
    pipe = load_model()  # Returns cached pipeline (loads once)

    # Use greedy decoding when temperature is very low (more stable)
    do_sample = temperature > 0.01

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=pipe.tokenizer.eos_token_id,
        eos_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=True,  # Return prompt + generated text
    )

    # The pipeline returns a list of dicts; we want the first result
    return outputs[0]["generated_text"]


# ── HuggingFace Inference API ─────────────────────────────────────────────────

def _generate_via_api(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Call the free HuggingFace Inference API to generate code.
    Useful if you don't have enough RAM/VRAM to run the model locally.

    Requires HUGGINGFACE_API_TOKEN in .env.

    Args:
        prompt:         The formatted prompt string.
        max_new_tokens: Token generation budget.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling p value.

    Returns:
        Raw generated text from the API.

    Raises:
        RuntimeError: On API errors (rate limit, bad token, model loading).
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "HUGGINGFACE_API_TOKEN is required for API mode. "
            "Set it in your .env file."
        )

    url = HF_API_URL.format(model=MODEL_NAME)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": True,
        },
        "options": {"wait_for_model": True},  # Wait if model is loading (cold start)
    }

    logger.info("Calling HuggingFace Inference API: %s", url)
    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace API returned HTTP {response.status_code}: {response.text}"
        )

    result = response.json()

    # API returns a list of generated text dicts
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "")

    raise RuntimeError(f"Unexpected API response format: {result}")
