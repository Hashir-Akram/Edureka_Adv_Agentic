"""Loads settings from the .env file. All secrets live in .env, never in code."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads .env from the project folder into environment variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

SYSTEM_PROMPT = (
    "You are a emma, you're helpFull assistant. "
    "If you are unsure about recent events, say so clearly."
)

if not GROQ_API_KEY or GROQ_API_KEY.startswith("paste_"):
    sys.exit("❌ GROQ_API_KEY missing. Open the .env file and paste your key.")