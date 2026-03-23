"""
app.py — Streamlit Web Interface for the AI Code Generator
===========================================================
Run with:
    streamlit run app.py

This file builds a professional, interactive UI that:
  - Accepts natural-language programming instructions
  - Shows generation settings in a collapsible sidebar
  - Displays generated code with full syntax highlighting
  - Provides copy-to-clipboard and download buttons
  - Shows model diagnostics in an expander
"""

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ─────────────────────────
st.set_page_config(
    page_title="AI Code Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import project modules (after set_page_config)
from code_generator import generate_code
from model_loader import get_model_info
from utils import validate_prompt, get_language_from_prompt

# ── Custom CSS for a modern, dark-themed look ───────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Header banner ── */
    .hero-banner {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    .hero-banner h1 { color: #fff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-banner p  { color: rgba(255,255,255,0.85); font-size: 1.05rem; margin: 0.4rem 0 0; }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Code output box ── */
    .stCodeBlock { border-radius: 10px !important; }

    /* ── Generate button ── */
    div.stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s, transform 0.1s;
    }
    div.stButton > button:hover  { opacity: 0.88; transform: translateY(-1px); }
    div.stButton > button:active { transform: translateY(0); }

    /* ── Sidebar labels ── */
    .css-1544g2n { color: #c8c8ff; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }
    .badge-green  { background: rgba(72,199,142,0.2); color: #48c78e; border: 1px solid #48c78e66; }
    .badge-purple { background: rgba(118,75,162,0.3); color: #b39ddb;  border: 1px solid #764ba277; }
    .badge-blue   { background: rgba(102,126,234,0.2); color: #90caf9; border: 1px solid #667eea66; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🤖 AI Code Generator</h1>
        <p>Describe what you want in plain English — get production-ready code instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — Settings & Info ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Generation Settings")

    max_tokens = st.slider(
        "Max New Tokens",
        min_value=50,
        max_value=800,
        value=300,
        step=50,
        help="Controls how long the generated code can be.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower = more deterministic. Higher = more creative.",
    )

    top_p = st.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Limits generation to the top-p probability mass.",
    )

    rep_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.05,
        help="Penalises repeated tokens to prevent looping output.",
    )

    auto_format = st.toggle(
        "Auto-format with Black",
        value=True,
        help="Automatically format generated Python code using Black.",
    )

    st.divider()

    # ── Model info ────────────────────────────────────────────────────────
    st.markdown("## 🔬 Model Info")
    try:
        info = get_model_info()
        st.markdown(
            f"""
            <span class="badge badge-purple">🧠 {info['model_name'].split('/')[-1]}</span>
            <span class="badge badge-blue">💻 {info['device'].upper()}</span>
            <span class="badge badge-green">{'🌐 API' if info['inference_mode'] == 'api' else '🏠 Local'}</span>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Full name: `{info['model_name']}`")
    except Exception:
        st.warning("Model info unavailable.")

    st.divider()

    # ── Example prompts ───────────────────────────────────────────────────
    st.markdown("## 💡 Example Prompts")
    EXAMPLES = [
        "Write Python code for bubble sort",
        "Write a Python function to reverse a string",
        "Create a Python class for a binary search tree",
        "Write a recursive function to compute Fibonacci numbers",
        "Build a Python function that reads a CSV file using pandas",
        "Write a decorator that measures function execution time",
        "Implement a stack data structure in Python",
        "Create a simple REST API with FastAPI that returns Hello World",
    ]
    for ex in EXAMPLES:
        if st.button(f"📌 {ex[:45]}…" if len(ex) > 45 else f"📌 {ex}", key=f"ex_{ex[:20]}"):
            st.session_state["prompt_input"] = ex


# ── Main Area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # ── Input section ─────────────────────────────────────────────────────
    st.markdown("### 📝 Your Instruction")

    # Restore prompt from session state (set by example buttons in sidebar)
    default_prompt = st.session_state.get("prompt_input", "")

    user_prompt = st.text_area(
        label="Describe the code you want:",
        value=default_prompt,
        height=140,
        placeholder=(
            "e.g. 'Write Python code for bubble sort'\n"
            "     'Create a function that checks if a number is prime'\n"
            "     'Implement a linked list in Python'"
        ),
        key="main_prompt_area",
        label_visibility="collapsed",
    )

    # Language override
    lang_col, _ = st.columns([1, 2])
    with lang_col:
        language_override = st.selectbox(
            "Language",
            options=["Auto-detect", "python", "javascript", "java", "cpp", "bash", "sql", "rust", "go"],
            index=0,
        )

    generate_btn = st.button("⚡ Generate Code", use_container_width=True)

    # ── Char counter & validation ─────────────────────────────────────────
    char_count = len(user_prompt)
    if char_count > 0:
        color = "#48c78e" if char_count <= 500 else "#ff6b6b"
        st.markdown(
            f'<p style="color:{color}; font-size:0.8rem; margin-top:-0.5rem;">'
            f"{char_count}/500 characters</p>",
            unsafe_allow_html=True,
        )

with col2:
    # ── Output section ────────────────────────────────────────────────────
    st.markdown("### 💻 Generated Code")

    # Placeholder while no code has been generated yet
    output_placeholder = st.empty()

    if "generated_result" not in st.session_state:
        output_placeholder.info(
            "👈  Type your instruction on the left and click **Generate Code**."
        )

# ── Generation logic (runs when button is clicked) ──────────────────────────
if generate_btn:
    # Validate input before hitting the model
    is_valid, error_msg = validate_prompt(user_prompt)

    if not is_valid:
        with col1:
            st.error(error_msg)
    else:
        language = None if language_override == "Auto-detect" else language_override

        # Show a progress spinner in col2
        with col2:
            output_placeholder.empty()
            with st.spinner("🔮 Generating code… this may take 15–60 seconds on first run."):
                result = generate_code(
                    user_instruction=user_prompt,
                    language=language,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    auto_format=auto_format,
                )

        # Store result in session state so it persists on re-runs
        st.session_state["generated_result"] = result

        # Clear the example-prompt session key
        if "prompt_input" in st.session_state:
            del st.session_state["prompt_input"]

# ── Display result if available ──────────────────────────────────────────────
if "generated_result" in st.session_state:
    result = st.session_state["generated_result"]

    with col2:
        output_placeholder.empty()

        if result.get("error"):
            st.error(f"❌ Generation failed:\n\n{result['error']}")
        elif result.get("code"):
            detected_lang = result.get("language", "python")

            # ── Syntax-highlighted code block ─────────────────────────
            st.code(result["code"], language=detected_lang)

            # ── Stats row ─────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Lines", result["code"].count("\n") + 1)
            m2.metric("Characters", len(result["code"]))
            m3.metric("Backend", result.get("backend", "local").upper())

            # ── Download button ────────────────────────────────────────
            ext_map = {
                "python": "py", "javascript": "js", "java": "java",
                "cpp": "cpp", "bash": "sh", "sql": "sql", "rust": "rs", "go": "go",
            }
            file_ext = ext_map.get(detected_lang, "txt")
            st.download_button(
                label="⬇️  Download Code",
                data=result["code"],
                file_name=f"generated_code.{file_ext}",
                mime="text/plain",
                use_container_width=True,
            )

            # ── Prompt used (in expander) ──────────────────────────────
            with st.expander("🔍 Show prompt sent to model"):
                st.code(result.get("prompt", ""), language="markdown")
        else:
            st.warning("The model returned no code. Try rephrasing your instruction.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align:center; color: rgba(255,255,255,0.4); font-size:0.8rem; padding: 0.5rem 0 1rem;'>
        AI Code Generator · Powered by HuggingFace Transformers · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
