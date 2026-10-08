"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-dangthevinh-2a202602587")
os.environ["LANGCHAIN_ENDPOINT"]   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# SDK mới đọc LANGSMITH_*; giữ đồng bộ với biến LANGCHAIN_* của đề bài.
for legacy, modern in {
    "LANGCHAIN_TRACING_V2": "LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY": "LANGSMITH_API_KEY",
    "LANGCHAIN_PROJECT": "LANGSMITH_PROJECT",
    "LANGCHAIN_ENDPOINT": "LANGSMITH_ENDPOINT",
}.items():
    os.environ[modern] = os.environ[legacy]

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-dangthevinh-2a202602587")



def validate() -> bool:
    """Validate provider keys, embedding fallback and tracing before execution."""
    missing = []
    def configured(value):
        return bool(value and not value.startswith("your_"))

    if not configured(LANGSMITH_API_KEY):
        missing.append("LANGCHAIN_API_KEY (LangSmith)")
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        missing.append("LANGCHAIN_TRACING_V2=true")
    keys = {
        "openai": ("OPENAI_API_KEY", OPENAI_API_KEY),
        "gemini": ("GOOGLE_API_KEY", GOOGLE_API_KEY),
        "anthropic": ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        "openrouter": ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
        "ollama": None,
    }
    if PROVIDER not in keys:
        missing.append("PROVIDER: openai/gemini/anthropic/openrouter/ollama")
    elif keys[PROVIDER] and not configured(keys[PROVIDER][1]):
        missing.append(keys[PROVIDER][0])
    if PROVIDER in ("anthropic", "openrouter") and not configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY (embeddings)")
    if missing:
        print("Missing or invalid configuration:")
        for item in missing:
            print(f"   - {item}")
        print("Fill .env in the project root; never commit API keys.")
        return False
    print(f"Config OK | Provider: {PROVIDER.upper()} | Project: {LANGSMITH_PROJECT}")
    return True

if __name__ == "__main__":
    raise SystemExit(0 if validate() else 1)
