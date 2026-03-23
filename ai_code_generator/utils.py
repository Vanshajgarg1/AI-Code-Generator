"""
utils.py — Helper functions for the AI Code Generator
======================================================
Contains reusable utilities for:
  - Prompt engineering (wrapping user instructions)
  - Code post-processing (cleaning model output)
  - Syntax highlighting for terminal display
  - Input validation
"""

import re
import logging
from pygments import highlight
from pygments.lexers import PythonLexer, get_lexer_by_name, ClassNotFound
from pygments.formatters import TerminalFormatter

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Prompt Engineering ────────────────────────────────────────────────────────

def build_prompt(user_instruction: str, language: str = "python") -> str:
    """
    Wrap the user's natural-language instruction into a structured prompt
    that guides the code model towards producing clean, runnable code.

    Args:
        user_instruction: Plain-English programming task from the user.
        language: Target programming language (default: "python").

    Returns:
        A formatted prompt string ready to be fed to the model.
    """
    user_instruction = user_instruction.strip()

    # Structured prompt — instruct-style format works well with most code LLMs
    prompt = (
        f"# Task: {user_instruction}\n"
        f"# Language: {language}\n"
        f"# Write a complete, well-commented {language} implementation below.\n\n"
    )
    logger.debug("Built prompt:\n%s", prompt)
    return prompt


# ── Code Post-Processing ──────────────────────────────────────────────────────

def extract_code_block(raw_output: str, prompt: str = "") -> str:
    """
    Clean and extract the generated code from the raw model output.

    Steps:
      1. Strip the echoed prompt from the output (models often repeat it).
      2. Extract a fenced markdown code block if one is present.
      3. Remove trailing incomplete lines (half-generated lines).
      4. Strip extra whitespace.

    Args:
        raw_output: The full decoded string returned by the model.
        prompt:     The original prompt (to strip the echo).

    Returns:
        A clean code string.
    """
    code = raw_output

    # 1. Remove the prompt echo if the model repeated it
    if prompt and code.startswith(prompt):
        code = code[len(prompt):]

    # 2. Try to extract a fenced code block  ```python ... ```
    fenced = re.search(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
    if fenced:
        code = fenced.group(1)

    # 3. Remove incomplete last line (no newline at end = truncated token)
    lines = code.splitlines()
    if lines and not lines[-1].endswith((":", ")", "]", "}", '"""', "'''")):
        # Keep the last line only if it looks complete
        last = lines[-1].strip()
        if last and not last.endswith(("return", "pass", "break", "continue")):
            # Heuristic: drop the last line if it seems cut off
            if len(last) > 0 and not last.endswith((":", '"', "'")):
                lines = lines[:-1]
    code = "\n".join(lines)

    # 4. Strip leading/trailing blank lines
    code = code.strip()

    return code


def format_with_black(code: str) -> str:
    """
    Auto-format Python code using the Black formatter.
    Falls back to the original string if Black raises an error.

    Args:
        code: Raw Python code string.

    Returns:
        Formatted Python code string.
    """
    try:
        import black  # imported lazily so the app still runs without black

        formatted = black.format_str(code, mode=black.Mode())
        logger.debug("Black formatting applied successfully.")
        return formatted
    except Exception as exc:
        logger.warning("Black formatting skipped: %s", exc)
        return code  # Return original code if formatting fails


def highlight_code(code: str, language: str = "python") -> str:
    """
    Apply terminal ANSI syntax highlighting to a code string using Pygments.
    Useful for CLI / debug output.

    Args:
        code:     Source code string.
        language: Programming language name for Pygments lexer lookup.

    Returns:
        ANSI-coloured string (safe to print in a terminal).
    """
    try:
        lexer = get_lexer_by_name(language)
    except ClassNotFound:
        lexer = PythonLexer()  # Fallback to Python lexer

    return highlight(code, lexer, TerminalFormatter())


# ── Input Validation ──────────────────────────────────────────────────────────

def validate_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate the user's input prompt before sending it to the model.

    Rules:
      - Must not be empty or whitespace-only.
      - Must be at least 5 characters long.
      - Must be at most 500 characters long.

    Args:
        prompt: Raw user input string.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
        If valid, error_message is an empty string.
    """
    if not prompt or not prompt.strip():
        return False, "⚠️  Prompt cannot be empty. Please describe what code you need."

    if len(prompt.strip()) < 5:
        return False, "⚠️  Prompt too short. Please be more descriptive."

    if len(prompt.strip()) > 500:
        return False, (
            f"⚠️  Prompt too long ({len(prompt.strip())} chars). "
            "Please keep it under 500 characters."
        )

    return True, ""


# ── Misc Utilities ────────────────────────────────────────────────────────────

def get_language_from_prompt(prompt: str) -> str:
    """
    Heuristically detect the target programming language from the prompt.
    Defaults to 'python' if no explicit language is found.

    Args:
        prompt: User instruction string.

    Returns:
        Language name as a lowercase string (e.g. 'python', 'javascript').
    """
    prompt_lower = prompt.lower()
    language_keywords = {
        "python": ["python", "py", "pandas", "numpy", "flask", "fastapi", "django"],
        "javascript": ["javascript", "js", "node", "react", "vue", "typescript"],
        "java": ["java", "spring", "maven"],
        "cpp": ["c++", "cpp"],
        "c": [" in c ", "c program"],
        "rust": ["rust"],
        "go": ["golang", " go "],
        "sql": ["sql", "query", "database", "select", "insert"],
        "bash": ["bash", "shell", "linux command"],
    }

    for language, keywords in language_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return language

    return "python"  # Default to Python


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate a long string for display purposes.

    Args:
        text:       Input string.
        max_length: Maximum number of characters.

    Returns:
        Truncated string with '…' appended if truncation occurred.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "…"
