import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    # Paths
    BASE_DIR = Path(__file__).parent
    CACHE_DIR = BASE_DIR / "cache"
    CHROMA_DIR = BASE_DIR / "knowledge_base"

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "papermind@example.com")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "groq")

    MODELS = {
        "groq": {
            "name": "llama-3.1-8b-instant",
            "temperature": 0.1,
            "max_tokens": 4096
        },
        "openai": {
            "name": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 4096
        },
        "anthropic": {
            "name": "claude-3-5-sonnet-20241022",
            "temperature": 0.1,
            "max_tokens": 4096
        },
        "google": {
            "name": "gemini-pro",
            "temperature": 0.1
        }
    }

    # Search settings
    PAPERS_PER_SOURCE = 5
    MAX_TOTAL_PAPERS = 20
    CACHE_RESULTS = os.getenv("CACHE_RESULTS", "true").lower() == "true"
    CACHE_EXPIRY_HOURS = 24

    # Create directories
    CACHE_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)


config = Config()
