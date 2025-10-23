# src/nodes/faq_node.py

from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.retrievers import RetrieverManager
from config import require_openai_key, LLM_MODEL
from constants import RECENT_MESSAGES_LIMIT, MAX_DOCS_FOR_FAQ
from typing import Optional, Type
from pydantic import BaseModel, Field

class FaqInput(BaseModel):
    """Input for FAQ tool."""
    question: str = Field(description="سؤال المستخدم حول الأسئلة المتكررة")

class FaqTool(BaseTool):
    """Tool for answering frequently asked questions."""
    name: str = "faq_tool"
    description: str = "للإجابة على الأسئلة المتكررة حول خدمات شركه متخصصه في الاتصالات، الشحن، الإلغاء، والاستفسارات العامة"
    args_schema: Type[BaseModel] = FaqInput

    def __init__(self, retriever_manager: RetrieverManager, memory: ConversationBufferMemory, model_name: str = LLM_MODEL):
        super().__init__()
        # Store components as private attributes to avoid Pydantic issues
        self._retriever = retriever_manager.retrievers.get("faq")
        if not self._retriever:
            raise ValueError("FAQ retriever مش متعرف")

        self._llm = ChatOpenAI(api_key=require_openai_key(), model_name=model_name, temperature=0)
        self._memory = memory

        self._prompt = ChatPromptTemplate.from_template("""
# دورك
أنت مساعد خدمة عملاء في شركة اتصالات.

# المعلومات المتاحة

## 🕐 تاريخ المحادثة:
{history}

## ❓ السؤال الجديد:
{question}

## 📚 قاعدة المعرفة (FAQs):
{context}

# طريقة الرد

1. **ابحث في قاعدة المعرفة** عن إجابة مباشرة
2. **إذا وجدت الإجابة:**
   - اذكرها بوضوح وإيجاز
   - أضف خطوات عملية إذا كانت إجرائية
   - أضف معلومة إضافية مفيدة إن وُجدت
3. **إذا لم تجد إجابة مباشرة:**
   - قل بوضوح: "عذراً، لا أملك إجابة دقيقة على هذا السؤال"
   - اقترح بديل: "لكن يمكنني مساعدتك في..."
   - أو: "يمكنك التواصل مع الدعم على: ١٦٠"

# أسلوب الرد
- ودود ومباشر
- لا تزيد عن 100 كلمة
- استخدم numbering للخطوات
- أضف emoji خفيف (مثل ✅ أو 📞)

# أمثلة

**مثال 1 - سؤال واضح:**
❓ كيف أشحن رصيد؟
✅ يمكنك شحن الرصيد بـ:
1. كروت الشحن من أي منفذ بيع
2. من خلال فودافون كاش
3. من خلال التطبيق

**مثال 2 - سؤال غامض:**
❓ عندي مشكلة
✅ عذراً، هل يمكنك توضيح المشكلة؟
- هل هي مشكلة في الشبكة؟
- أم في الفاتورة؟
- أم شيء آخر؟

الإجابة:
""")

    def _run(self, question: str, session_id: Optional[str] = None) -> str:
        """Run the FAQ tool."""
        docs = self._retriever.get_relevant_documents(question)
        if not docs:
            response = "عذرًا، لم أجد إجابة على سؤالك. يمكنك التواصل مع الدعم على: ١٦٠"
        else:
            context = "\n".join([d.page_content for d in docs[:MAX_DOCS_FOR_FAQ]])
            
            # Get chat history from the agent's memory
            history_text = ""
            if self._memory and hasattr(self._memory, 'chat_memory'):
                messages = self._memory.chat_memory.messages
                # Get last few messages for context
                recent_messages = messages[-RECENT_MESSAGES_LIMIT:] if len(messages) > RECENT_MESSAGES_LIMIT else messages
                history_text = "\n".join([
                    f"{'User' if hasattr(msg, 'content') and 'Human' in str(type(msg)) else 'Assistant'}: {msg.content}" 
                    for msg in recent_messages if hasattr(msg, 'content')
                ])
            
            chain_input = {
                "question": question, 
                "context": context,
                "history": history_text
            }
            response = self._llm.predict(self._prompt.format(**chain_input)).strip()

        return response

    async def _arun(self, question: str, session_id: Optional[str] = None) -> str:
        """Async run method."""
        return self._run(question, session_id)

