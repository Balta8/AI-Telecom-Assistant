import re
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from utils.retrievers import RetrieverManager
from src.nodes.faq_node import FaqTool
from src.nodes.package_info_node import PackageInfoTool
from src.nodes.package_recommendation_node import PackageRecommendationTool
from src.nodes.support_node import SupportTool
from config import require_openai_key, LLM_MODEL
from constants import (
    EMPTY_MESSAGE, MESSAGE_TOO_LONG, MESSAGE_TOO_SHORT, PROCESSING_ERROR,
    NO_RESPONSE, MAX_MESSAGE_LENGTH, MIN_MESSAGE_LENGTH, MAX_ACTIVE_SESSIONS,
    AGENT_MAX_ITERATIONS, AGENT_TEMPERATURE, AGENT_REQUEST_TIMEOUT, AGENT_MAX_RETRIES
)
import re

class CustomerSupportAgent:
    """
    React Agent for customer support with per-session memory.
    """

    def __init__(self, retriever_manager: RetrieverManager):
        retriever_manager.setup_retrievers()
        self.retriever_manager = retriever_manager
        self.llm = ChatOpenAI(
            api_key=require_openai_key(),
            model_name=LLM_MODEL,
            temperature=AGENT_TEMPERATURE,
            request_timeout=AGENT_REQUEST_TIMEOUT,
            max_retries=AGENT_MAX_RETRIES
        )
        # dictionary for active agents per session
        self.sessions = {}

    def _create_agent_for_session(self, session_id: str):
        """Create a new agent with its own memory for a specific session."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        system_prompt = """
# دورك
أنت مساعد ذكي لخدمة عملاء شركة النمسا المتحدة للاتصالات.
اسمك: "مساعد فودافون الذكي" ✨

# المهام الرئيسية
1. 🔍 الإجابة على الأسئلة المتكررة (FAQ)
2. 📦 تقديم معلومات مفصلة ودقيقة عن الباقات
3. 🎯 ترشيح الباقة الأمثل بناءً على احتياجات العميل
4. 🛠️ حل المشاكل التقنية والإدارية
5. 🔄 مقارنة الباقات عند الطلب

# شخصيتك
- ودود، مهني، وصبور
- تستخدم العربية الفصحى المبسطة
- تضيف emojis بشكل خفيف لتحسين التجربة 😊
- تتذكر تفاصيل المحادثة السابقة

# قواعد صارمة للمعلومات 🚫
- لا تخترع أي معلومات غير موجودة في قاعدة البيانات
- لا تفترض أسعار أو تفاصيل غير مذكورة
- إذا لم تجد المعلومة، قل بوضوح: "عذراً، لا أملك هذه المعلومة حالياً"
- لا تقترح باقات لم تُذكر في البيانات المُسترجعة

# أسلوب الرد
✅ الرد المثالي:
- مباشر وواضح
- منظم (points أو numbering عند الحاجة)
- يجيب على السؤال بالضبط
- لا يحتوي على معلومات زائدة

❌ تجنب:
- الردود الطويلة بدون داعٍ
- التكرار
- المعلومات العامة غير المفيدة

# استخدام الأدوات
📌 faq_tool:
- لأسئلة الشحن، الإلغاء، الخدمات العامة
- مثال: "كيف أشحن رصيد؟"

📌 package_info_tool:
- للحصول على تفاصيل باقة محددة بالاسم
- مثال: "تفاصيل فليكس ٧٠"
- لا تستخدمها لعرض كل الباقات

📌 package_recommendation_tool:
- لترشيح باقة بناءً على الاحتياجات
- لعرض قائمة كل الباقات المتاحة
- مثال: "باقة للمكالمات بحد ١٠٠ج" أو "ايه الباقات المتاحة؟"

📌 support_tool:
- للمشاكل التقنية والإدارية
- مثال: "مشكلة في الراوتر"

# المقارنة بين الباقات 🔄
عند طلب المقارنة:
1. استخدم package_info_tool لكل باقة
2. اعرض مقارنة واضحة:
   
   📊 المقارنة:
   
   🔹 [الباقة الأولى]:
   - السعر: ...
   - المميزات: ...
   
   🔹 [الباقة الثانية]:
   - السعر: ...
   - المميزات: ...
   
   ✅ التوصية: ... (مع ذكر السبب)

# الذاكرة والسياق 🧠
- تذكر الباقات المذكورة سابقاً في المحادثة
- عند الإشارة لـ "الباقة السابقة" أو "الأولى"، استخدم الذاكرة
- اربط المعلومات الجديدة بالسياق السابق

# التعامل مع الحالات الخاصة
❓ إذا كان السؤال غامضاً:
"عذراً، هل يمكنك توضيح سؤالك أكثر؟ مثلاً: هل تريد باقة للإنترنت أم للمكالمات؟"

🔍 إذا لم تجد الباقة:
"عذراً، لم أجد باقة بهذا الاسم. هل تريد معرفة الباقات المتاحة؟"

🚀 إذا احتاج تدخل بشري:
"سأقوم بتحويلك لفريق الدعم المتخصص. رقم الدعم: ١٦٠ (متاح ٢٤/٧)"
"""
        # Add system prompt to memory
        memory.chat_memory.add_message(SystemMessage(content=system_prompt))

        # Initialize tools with session memory
        tools = [
            FaqTool(self.retriever_manager, memory),
            PackageInfoTool(self.retriever_manager, memory),
            PackageRecommendationTool(self.retriever_manager, memory),
            SupportTool(memory)
        ]

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=self._handle_parsing_error,
            max_iterations=AGENT_MAX_ITERATIONS,
            early_stopping_method="generate"
        )
        self.sessions[session_id] = agent
        return agent

    def _handle_parsing_error(self, error_message: str) -> str:
        """Custom handler for parsing errors - extract the actual response"""
        # Extract the content between backticks or after "Could not parse LLM output:"
        if "Could not parse LLM output:" in error_message:
            # Find the content after the error message
            content_start = error_message.find("`") + 1
            content_end = error_message.rfind("`")
            if content_start > 0 and content_end > content_start:
                content = error_message[content_start:content_end]
                # Clean any remaining backticks or formatting
                content = content.replace("```", "").strip()
                return content
        
        # If we can't extract content, return the original but cleaned
        cleaned = error_message.replace("Could not parse LLM output:", "")
        cleaned = cleaned.replace("`", "").replace("For troubleshooting, visit:", "")
        cleaned = cleaned.split("https://python.langchain.com")[0]  # Remove the URL
        return cleaned.strip()

    def handle_message(self, session_id: str, user_message: str) -> str:
        """Handle user message with a session-specific agent."""
        try:
            # Input validation
            if not user_message or not user_message.strip():
                return EMPTY_MESSAGE
            if len(user_message) > MAX_MESSAGE_LENGTH:
                return MESSAGE_TOO_LONG
            if len(user_message.strip()) < MIN_MESSAGE_LENGTH:
                return MESSAGE_TOO_SHORT
            
            # Get or create agent for this session
            agent = self.sessions.get(session_id)
            if not agent:
                agent = self._create_agent_for_session(session_id)
            
            # Cleanup old sessions if too many
            if len(self.sessions) > MAX_ACTIVE_SESSIONS:
                oldest_session = list(self.sessions.keys())[0]
                del self.sessions[oldest_session]

            # Run the agent - memory is handled automatically by LangChain
            response = agent.run(user_message)
            
            # Clean the response
            response = self._clean_response(response)
            
            return response

        except Exception as e:
            # Handle specific parsing errors
            error_str = str(e)
            if "OutputParserException" in error_str:
                # Try to extract useful information from the error
                if "is not iterable" in error_str:
                    return "عذراً، حدث خطأ في معالجة طلبك. يرجى إعادة صياغة سؤالك بشكل أوضح."
                else:
                    return "عذراً، حدث خطأ في فهم طلبك. هل يمكنك إعادة صياغة سؤالك؟"
            else:
                return PROCESSING_ERROR.format(error=error_str)
    
    def _clean_response(self, response: str) -> str:
        """Clean the response from unwanted strings and artifacts"""
        if not response:
            return NO_RESPONSE
        
        # Remove common parsing errors
        response = response.replace("undefined", "").replace("null", "")
        
        # Remove markdown artifacts
        response = response.replace("```json", "").replace("```", "")
        
        # Remove agent internal thoughts/actions if they leaked through
        response = re.sub(r'Action:.*?\n', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Thought:.*?\n', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Observation:.*?\n', '', response, flags=re.IGNORECASE)
        response = re.sub(r'AI:.*?\n', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Final Answer:.*?\n', '', response, flags=re.IGNORECASE)
        
        # Remove any "Input:" or "Output:" labels that might leak
        response = re.sub(r'^(Input|Output):\s*', '', response, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean extra whitespace but preserve line breaks
        lines = response.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines if line.strip()]
        response = '\n'.join(cleaned_lines)
        
        # Remove any repeated emojis (more than 3 in a row)
        response = re.sub(r'([\U0001F300-\U0001F9FF])\1{3,}', r'\1\1\1', response)
        
        return response.strip()


def create_agent(retriever_manager: RetrieverManager) -> CustomerSupportAgent:
    """Factory helper used by the app to create a CustomerSupportAgent instance.

    Keeps the top-level import used in `app.py` simple: `from agent import create_agent`.
    """
    return CustomerSupportAgent(retriever_manager)
