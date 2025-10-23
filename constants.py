# constants.py
"""
Constants and templates for the Vodafone chatbot
"""

# ===== Response Templates =====

PACKAGE_NOT_FOUND = "عذراً، لم أجد باقة بهذا الاسم في قاعدة البيانات. يرجى التأكد من اسم الباقة."

NO_INFORMATION = "عذراً، لا أملك معلومات كافية حول هذا الموضوع."

EMPTY_MESSAGE = "عذراً، لم أستلم أي رسالة. هل يمكنك إعادة المحاولة؟"

MESSAGE_TOO_LONG = "عذراً، رسالتك طويلة جداً. يرجى اختصارها أو تقسيمها لعدة رسائل."

MESSAGE_TOO_SHORT = "عذراً، رسالتك قصيرة جداً. هل يمكنك توضيح ما تريده؟"

PROCESSING_ERROR = "عذراً، حدث خطأ في المعالجة: {error}"

NO_RESPONSE = "عذراً، لم أتمكن من تكوين رد مناسب."

ESCALATION_MESSAGE = """
🙋 سأحولك للدعم المتخصص للمساعدة بشكل أفضل.

📞 **رقم الدعم:** ١٦٠ (متاح ٢٤/٧)
💬 **الدردشة المباشرة:** متاحة عبر التطبيق
"""

# ===== Listing Keywords =====

LISTING_KEYWORDS = [
    "كل الباقات", "الباقات المتاحة", "ايه الباقات", "اية الباقات",
    "الباقات الموجودة", "عايز اعرف الباقات", "شوف الباقات",
    "اعرض الباقات", "الباقات الي عندكو", "عندكم ايه",
    "الباقات اللي عندكم", "شوفني الباقات",
    "all packages", "available packages", "show packages"
]

# ===== Validation Settings =====

MAX_MESSAGE_LENGTH = 1000
MIN_MESSAGE_LENGTH = 2
MAX_ACTIVE_SESSIONS = 100

# ===== Retrieval Settings =====

MAX_DOCS_FOR_RECOMMENDATION = 6
MAX_DOCS_FOR_LISTING = 15
MAX_DOCS_FOR_FAQ = 5
MAX_DOCS_PER_CATEGORY = 4

# ===== Search Queries for Diverse Listing =====

DIVERSE_PACKAGE_QUERIES = ["فليكس", "plus", "باقة انترنت", "باقة مكالمات"]

# ===== Agent Settings =====

AGENT_MAX_ITERATIONS = 5
AGENT_TEMPERATURE = 0.3
AGENT_REQUEST_TIMEOUT = 30
AGENT_MAX_RETRIES = 2

# ===== Memory Settings =====

RECENT_MESSAGES_LIMIT = 6
SUPPORT_HISTORY_LIMIT = 8

