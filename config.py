import os
from dotenv import load_dotenv

load_dotenv()

# الإعدادات الأساسية
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def require_openai_key():
    """Validates that OpenAI API key is available. Call this before using LLM."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your environment or .env file."
        )
    return OPENAI_API_KEY
# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "vodafone_packages"

# إعدادات الـ Agent
AGENT_TEMPERATURE = 0.3
MAX_QUESTIONS = 4  # عدد الأسئلة القصوى في الحوار الاستشاري
