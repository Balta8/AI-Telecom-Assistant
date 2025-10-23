"""
Chainlit Chat Interface for Vodafone Customer Support
"""
import chainlit as cl
from agent import CustomerSupportAgent
from utils.retrievers import RetrieverManager
import uuid

# Global agent instance
agent = None
retriever_manager = None

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    global agent, retriever_manager
    
    # Initialize agent if not already done
    if agent is None:
        await cl.Message(
            content="🤖 جاري تهيئة المساعد الذكي...",
        ).send()
        
        retriever_manager = RetrieverManager(persist_directory="./chroma_store")
        agent = CustomerSupportAgent(retriever_manager)
        
        await cl.Message(
            content="✅ تم تهيئة المساعد بنجاح!",
        ).send()
    
    # Generate a unique session ID for this user
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Send welcome message
    welcome_message = """
# 👋 مرحباً بك في النمسا المتحدة للاتصالات!

أنا مساعدك الذكي، يمكنني مساعدتك في:

📋 **الأسئلة المتكررة** - الإجابة على استفساراتك العامة
📦 **معلومات الباقات** - تفاصيل كاملة عن الباقات المتاحة  
🎯 **ترشيح الباقات** - إيجاد الباقة المثالية لك
🛠️ **الدعم الفني** - حل المشاكل التقنية

💬 اكتب رسالتك وسأكون سعيداً بمساعدتك!
    """
    
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    global agent
    
    # Get session ID
    session_id = cl.user_session.get("session_id")
    
    # Show typing indicator
    async with cl.Step(name="🤖 جاري التفكير...") as step:
        try:
            # Get response from agent
            response = agent.handle_message(session_id, message.content)
            
            step.output = "✅ تم الحصول على الرد"
        except Exception as e:
            response = f"عذراً، حدث خطأ: {str(e)}"
            step.output = "❌ حدث خطأ"
    
    # Send response
    await cl.Message(
        content=response,
    ).send()


@cl.on_chat_end
def end():
    """Handle chat end"""
    print("Chat session ended")


# Custom settings
@cl.set_chat_profiles
async def chat_profile():
    """Define chat profiles"""
    return [
        cl.ChatProfile(
            name="Arabic Support",
            markdown_description="مساعد خدمة العملاء بالعربية 🇪🇬",
            icon="https://picsum.photos/200",
        ),
    ]

