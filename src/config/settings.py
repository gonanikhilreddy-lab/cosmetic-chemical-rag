from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CSV_PATH = RAW_DATA_DIR / "interviewtestdataset.csv"
DATABASE_PATH = PROCESSED_DATA_DIR / "cosmetics.duckdb"
VECTORSTORE_PATH = PROJECT_ROOT / os.getenv("VECTORSTORE_PATH", "vectorstore/chroma")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

load_dotenv(PROJECT_ROOT / ".env")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
VECTORSTORE_PATH = PROJECT_ROOT / os.getenv("VECTORSTORE_PATH", str(VECTORSTORE_PATH.relative_to(PROJECT_ROOT)))