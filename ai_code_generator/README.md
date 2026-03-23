# 🤖 AI Code Generator

> **GitHub Copilot-like AI tool** — type a plain-English instruction, get production-ready code instantly.  
> Powered by **HuggingFace Transformers** · Built with **Streamlit**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 Code LLM | CodeGen-350M (default), StarCoder2, CodeLlama, Phi-2, … |
| 🖥️ UI | Streamlit dark-themed dashboard with syntax highlighting |
| 📥 Download | Save generated code as `.py`, `.js`, `.java`, … |
| 🎛️ Controls | Temp, Top-p, max tokens via sidebar sliders |
| 🔌 Backends | Local (CPU/GPU) **or** HuggingFace Inference API |
| 🌍 Languages | Python, JavaScript, Java, C++, Rust, Go, SQL, Bash |
| ⚫ Formatting | Auto-format Python with Black |

---

## 🗂️ Project Structure

```
ai_code_generator/
│
├── app.py              ← Streamlit web interface
├── model_loader.py     ← HuggingFace model download & caching
├── code_generator.py   ← Core: prompt → generated code
├── utils.py            ← Prompt building, validation, post-processing
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
└── README.md           ← This file
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)               │
│  ┌─────────────────┐          ┌──────────────────────┐  │
│  │  Sidebar config │          │  Code output block   │  │
│  │  (sliders,      │          │  (syntax highlight,  │  │
│  │   examples)     │          │   download button)   │  │
│  └────────┬────────┘          └──────────┬───────────┘  │
│           │  user prompt                  │ generated code│
└───────────┼───────────────────────────────┼──────────────┘
            ▼                               │
┌───────────────────────┐                  │
│   code_generator.py   │──────────────────┘
│  1. validate_prompt() │
│  2. build_prompt()    │
│  3. call model/API    │
│  4. extract_code()    │
│  5. format w/ Black   │
└───────┬───────────────┘
        │
   ┌────┴───────────────────────────────────┐
   │                                         │
   ▼  (INFERENCE_MODE=local)                 ▼  (INFERENCE_MODE=api)
┌─────────────────────┐          ┌─────────────────────────┐
│   model_loader.py   │          │  HuggingFace Inference   │
│  AutoTokenizer      │          │  API (free, no GPU)      │
│  AutoModelForCausal │          │  api-inference.hf.co     │
│  LM  (cached)       │          └─────────────────────────┘
└─────────────────────┘
```

---

## ⚡ Quick Start

### 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd ai_code_generator
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads ~350 MB PyTorch + Transformers.  
> The model weights (~350 MB for `codegen-350M-mono`) are downloaded on first launch.

### 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your values:

```env
# Required only for gated models (e.g. CodeLlama) or API mode
HUGGINGFACE_API_TOKEN=hf_your_token_here

# Model to use (lightweight default — works on CPU in <2 GB RAM)
MODEL_NAME=Salesforce/codegen-350M-mono

# "local" runs on your machine; "api" calls HuggingFace for free
INFERENCE_MODE=local
```

### 5 — Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🚀

---

## 🧠 Model Options

| Model | Size | Quality | RAM needed | Notes |
|---|---|---|---|---|
| `Salesforce/codegen-350M-mono` | 350 MB | ⭐⭐⭐ | ~2 GB | **Default · CPU-friendly** |
| `bigcode/starcoder2-3b` | 3 B | ⭐⭐⭐⭐ | ~6 GB | Great quality |
| `microsoft/phi-2` | 2.7 B | ⭐⭐⭐⭐ | ~6 GB | General + code |
| `meta-llama/CodeLlama-7b-hf` | 7 B | ⭐⭐⭐⭐⭐ | ~14 GB | Needs HF token (gated) |
| `bigcode/starcoder2-7b` | 7 B | ⭐⭐⭐⭐⭐ | ~14 GB | Needs GPU |

Change the model in `.env`:
```env
MODEL_NAME=bigcode/starcoder2-3b
```

---

## 💡 Example Prompts

```
Write Python code for bubble sort
```
```
Write a Python function to reverse a string
```
```
Create a Python class for a binary search tree with insert and search
```
```
Write a recursive function to compute Fibonacci numbers
```
```
Build a Python function that reads a CSV file and returns a pandas DataFrame
```
```
Write a Python decorator that measures and logs function execution time
```
```
Implement a stack data structure in Python using a list
```
```
Create a simple FastAPI REST API that returns Hello World on /
```
```
Write a Python function that checks if a string is a palindrome
```
```
Implement merge sort in Python with step-by-step comments
```

---

## 🎛️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Salesforce/codegen-350M-mono` | HuggingFace model ID |
| `HUGGINGFACE_API_TOKEN` | _(empty)_ | Required for gated models or API mode |
| `INFERENCE_MODE` | `local` | `local` or `api` |
| `MAX_NEW_TOKENS` | `300` | Max tokens to generate |
| `TEMPERATURE` | `0.2` | Creativeness (0 = deterministic) |
| `TOP_P` | `0.95` | Nucleus sampling |
| `REPETITION_PENALTY` | `1.1` | Discourages repeated tokens |
| `FORCE_CPU` | `false` | Force CPU even if GPU is available |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| **Out of memory** | Switch to `codegen-350M-mono` or set `FORCE_CPU=true` |
| **Slow generation** | Normal on CPU; expect 15–60 s. Use GPU or API mode for speed. |
| **401 Unauthorized** | Set `HUGGINGFACE_API_TOKEN` in `.env` |
| **Model not found** | Check spelling; some models need access request on HF |
| **Black format error** | Toggle off "Auto-format with Black" in the sidebar |
| **Port in use** | Run `streamlit run app.py --server.port 8502` |

---

## 🔮 Optional: FastAPI Backend

To expose a REST API alongside the Streamlit UI:

```bash
pip install fastapi uvicorn
```

Create `api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from code_generator import generate_code

app = FastAPI(title="AI Code Generator API")

class GenerateRequest(BaseModel):
    instruction: str
    language: str = None
    max_new_tokens: int = 300
    temperature: float = 0.2

@app.post("/generate")
def generate(req: GenerateRequest):
    return generate_code(
        user_instruction=req.instruction,
        language=req.language,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
```

Run:
```bash
uvicorn api:app --reload
# Docs at http://localhost:8000/docs
```

---

## 📄 License

MIT — free to use, modify, and distribute.
