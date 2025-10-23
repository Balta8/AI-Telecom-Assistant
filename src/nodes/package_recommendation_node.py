# src/nodes/package_recommendation_node.py

from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from utils.retrievers import RetrieverManager
from config import require_openai_key, LLM_MODEL
from constants import (
    LISTING_KEYWORDS, MAX_DOCS_FOR_RECOMMENDATION, MAX_DOCS_FOR_LISTING,
    DIVERSE_PACKAGE_QUERIES, MAX_DOCS_PER_CATEGORY, RECENT_MESSAGES_LIMIT
)
from typing import Optional, Type
from pydantic import BaseModel, Field

class PackageRecommendationInput(BaseModel):
    """Input for package recommendation tool."""
    user_needs: str = Field(description="احتياجات المستخدم وتفضيلاته لترشيح الباقة")

class PackageRecommendationTool(BaseTool):
    """Tool for recommending the best package based on user needs."""
    name: str = "package_recommendation_tool"
    description: str = """
استخدم هذه الأداة لترشيح باقة أو عرض كل الباقات المتاحة.

متى تستخدم هذه الأداة:
✅ "عايز باقة للمكالمات بحد ١٠٠ج"
✅ "رشحلي باقة مناسبة"
✅ "ايه كل الباقات المتاحة؟"
✅ "عايز اعرف الباقات الموجودة"

متى لا تستخدمها:
❌ "تفاصيل فليكس ٧٠" → استخدم package_info_tool
❌ "معلومات عن Plus 155" → استخدم package_info_tool

المدخل المطلوب: احتياجات العميل أو طلب عرض كل الباقات
"""
    args_schema: Type[BaseModel] = PackageRecommendationInput

    def __init__(self, retriever_manager: RetrieverManager, memory: ConversationBufferMemory):
        super().__init__()
        self._retriever_manager = retriever_manager
        self._memory = memory
        self._llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3, api_key=require_openai_key())

        self._prompt = PromptTemplate(
            input_variables=["query", "docs", "preferences", "history"],
            template="""
# دورك
أنت مساعد ذكي في شركة اتصالات. مهمتك: ترشيح الباقة المناسبة أو عرض الباقات المتاحة.

# المعلومات المتاحة

## 📝 طلب العميل:
{query}

## 📦 الباقات المتوفرة (استخدم فقط هذه القائمة):
{docs}

## 🕐 سياق المحادثة السابقة:
{history}

## 🎯 تفضيلات العميل (إن وُجدت):
{preferences}

# طريقة العمل

## إذا طلب "كل الباقات" أو "الباقات المتاحة":
اعرض القائمة بشكل منظم:

📦 **باقات فليكس:**
• فليكس ٧٠ — [التفاصيل] — السعر: [السعر]
• فليكس ١٠٠ — [التفاصيل] — السعر: [السعر]

📦 **باقات Plus:**
• Plus 155 — [التفاصيل] — السعر: [السعر]

💡 هل تريد تفاصيل أكثر عن باقة معينة؟

## إذا طلب ترشيح باقة:

### خطوة 1: تحليل الاحتياجات
- الميزانية: [هل ذكرها؟]
- الاستخدام الأساسي: [إنترنت / مكالمات / مزيج]
- أي تفضيلات أخرى من السياق

### خطوة 2: الترشيح
اختر الباقة الأنسب من القائمة أعلاه

### خطوة 3: اعرض التوصية

✅ **الباقة الموصى بها:**
📦 [اسم الباقة]

**لماذا؟**
• [سبب 1]
• [سبب 2]

**السعر:** [السعر]
**المميزات:** [قائمة بالمميزات]

💡 هل تريد المقارنة مع باقة أخرى؟

# قواعد صارمة 🚫
1. استخدم فقط الباقات الموجودة في قسم "الباقات المتوفرة" أعلاه
2. لا تخترع أسعار أو مميزات غير موجودة
3. إذا كانت المعلومات ناقصة، اسأل سؤال واحد محدد فقط
4. لا تذكر أكثر من 3 باقات في الترشيح
5. رتب الباقات من الأنسب للأقل مناسبة

# أسلوب الرد
- واضح ومباشر
- منظم (bullets أو numbering)
- يحتوي على emoji بسيط
- لا يزيد عن 200 كلمة إلا إذا طُلبت كل الباقات

الإجابة:
"""
        )
        self._chain = LLMChain(llm=self._llm, prompt=self._prompt)

    def _run(self, user_needs: str, session_id: Optional[str] = None) -> str:
        """Run the package recommendation tool."""
        # Check if user is asking for all packages
        query_lower = user_needs.lower()
        is_listing_request = any(word in query_lower for word in LISTING_KEYWORDS)
        
        if is_listing_request:
            # Get diverse packages for listing
            all_docs = []
            
            # Strategy: Get diverse packages by querying different terms
            for query in DIVERSE_PACKAGE_QUERIES:
                docs = self._retriever_manager.get_documents(query, "package")
                all_docs.extend(docs[:MAX_DOCS_PER_CATEGORY])
            
            # Remove duplicates based on content
            seen = set()
            unique_docs = []
            for doc in all_docs:
                content = doc.page_content
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(doc)
            
            # Format with better structure (numbered list)
            docs_text = "\n".join([f"{i+1}. {doc.page_content}" 
                                  for i, doc in enumerate(unique_docs[:MAX_DOCS_FOR_LISTING])])
        else:
            # Get specific recommendations
            docs = self._retriever_manager.get_documents(user_needs, "package")
            if not docs:
                return "عذراً، لم أجد باقات مناسبة لاحتياجاتك. هل يمكنك توضيح متطلباتك أكثر؟"
            
            docs_text = "\n".join([f"- {doc.page_content}" 
                                  for doc in docs[:MAX_DOCS_FOR_RECOMMENDATION]])

        # Get chat history from the agent's memory
        history_text = ""
        if self._memory and hasattr(self._memory, 'chat_memory'):
            messages = self._memory.chat_memory.messages
            recent_messages = messages[-RECENT_MESSAGES_LIMIT:] if len(messages) > RECENT_MESSAGES_LIMIT else messages
            history_text = "\n".join([
                f"{'User' if hasattr(msg, 'content') and 'Human' in str(type(msg)) else 'Assistant'}: {msg.content}" 
                for msg in recent_messages if hasattr(msg, 'content')
            ])

        response = self._chain.run(
            query=user_needs,
            docs=docs_text,
            preferences="",  # We'll get preferences from conversation context
            history=history_text
        )

        return response

    async def _arun(self, user_needs: str, session_id: Optional[str] = None) -> str:
        """Async run method."""
        return self._run(user_needs, session_id)

