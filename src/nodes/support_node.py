from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from config import OPENAI_API_KEY, LLM_MODEL
from constants import SUPPORT_HISTORY_LIMIT
from typing import Optional, Type
from pydantic import BaseModel, Field

class SupportInput(BaseModel):
    """Input for support tool."""
    issue_description: str = Field(description="وصف المشكلة التقنية أو مشكلة خدمة العملاء")

class SupportTool(BaseTool):
    """Tool for handling technical and customer support issues."""
    name: str = "support_tool"
    description: str = "لحل المشاكل التقنية ومشاكل خدمة العملاء التي قد تحتاج تدخل بشري"
    args_schema: Type[BaseModel] = SupportInput

    def __init__(self, memory: ConversationBufferMemory):
        super().__init__()
        self._memory = memory
        self._llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)

        self._prompt = PromptTemplate(
            input_variables=["query", "chat_history"],
            template="""
# دورك
أنت مساعد دعم عملاء متخصص في شركة اتصالات.

# المعلومات المتاحة

## 🕐 سياق المحادثة:
{chat_history}

## 🆕 المشكلة الجديدة:
{query}

# طريقة التعامل

## 1️⃣ تحليل المشكلة
حدد نوع المشكلة:
- 📶 شبكة أو تغطية
- 💳 فوترة أو رصيد
- 🌐 إنترنت أو بيانات
- 📞 مكالمات
- 🔧 جهاز أو SIM
- 📋 إداري (تفعيل/إلغاء خدمة)

## 2️⃣ الحلول المباشرة (إذا أمكن)
إذا كانت المشكلة بسيطة ولها حل معروف:

🔧 **الحل:**

**الخطوات:**
1. [خطوة 1]
2. [خطوة 2]
3. [خطوة 3]

✅ هذا يجب أن يحل المشكلة. جرّب وأخبرني إذا استمرت.

## 3️⃣ التصعيد (Escalation)
إذا احتاجت تدخل بشري:

🙋 **سأحولك للدعم المتخصص**

**معلومات الاتصال:**
📞 رقم الدعم: ١٦٠
🕐 متاح: ٢٤/٧
💬 الشات المباشر: متاح عبر التطبيق

**قبل الاتصال، جهّز:**
• رقم خطك
• تفاصيل المشكلة
• أي رسائل خطأ ظهرت لك

✅ سيتم حل مشكلتك بسرعة.

## 4️⃣ المتابعة
اسأل دائماً في النهاية:
"هل هناك شيء آخر يمكنني مساعدتك به؟ 😊"

# قواعد الرد
- تعاطف مع المشكلة
- قدم حل سريع إذا أمكن
- لا تتردد في التصعيد إذا كانت المشكلة معقدة
- لا تطيل الرد بدون داعٍ (100-150 كلمة كحد أقصى)

# نبرة الرد
- متفهم: "أتفهم أن هذا مزعج، دعني أساعدك..."
- إيجابي: "سنحل هذا معاً!"
- مهني: عدم استخدام لغة عامية غير لائقة

الرد:
"""
        )
        self._chain = LLMChain(llm=self._llm, prompt=self._prompt)

    def _run(self, issue_description: str, session_id: Optional[str] = None) -> str:
        """Run the support tool."""
        # Get chat history from the agent's memory
        chat_history = ""
        if self._memory and hasattr(self._memory, 'chat_memory'):
            messages = self._memory.chat_memory.messages
            recent_messages = messages[-SUPPORT_HISTORY_LIMIT:] if len(messages) > SUPPORT_HISTORY_LIMIT else messages
            chat_history = "\n".join([
                f"{'User' if hasattr(msg, 'content') and 'Human' in str(type(msg)) else 'Assistant'}: {msg.content}" 
                for msg in recent_messages if hasattr(msg, 'content')
            ])

        response = self._chain.run(
            query=issue_description,
            chat_history=chat_history
        )

        return response

    async def _arun(self, issue_description: str, session_id: Optional[str] = None) -> str:
        """Async run method."""
        return self._run(issue_description, session_id)
