#!/usr/bin/env python3
"""
Vodafone Customer Support Chatbot
Main application entry point
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent import VodafoneAgent
from utils.retrievers import RetrieverManager

def main():
    """Main application entry point"""
    print("🚀 بدء تشغيل مساعد فودافون الذكي...")
    
    try:
        # Initialize components
        print("⚙️  تهيئة المكونات...")
        retriever_manager = RetrieverManager(persist_directory="./chroma_store")
        bot = VodafoneAgent(retriever_manager)
        
        print("✅ تم تهيئة البوت بنجاح!")
        print("\n📱 يمكنك الآن استخدام البوت:")
        print("   - Streamlit: streamlit run streamlit_app.py")
        print("   - Chainlit: chainlit run chainlit_app.py --port 8000")
        print("   - أو استخدم البوت مباشرة من الكود")
        
        # Interactive mode
        print("\n💬 وضع التفاعل المباشر:")
        print("اكتب 'exit' للخروج")
        print("-" * 50)
        
        session_id = "interactive_session"
        
        while True:
            try:
                user_input = input("\n👤 أنت: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'خروج']:
                    print("👋 وداعاً!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 البوت: ", end="")
                response = bot.handle_message(session_id, user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 وداعاً!")
                break
            except Exception as e:
                print(f"❌ خطأ: {e}")
                
    except Exception as e:
        print(f"❌ فشل في تهيئة البوت: {e}")
        print("💡 تأكد من:")
        print("   - وجود ملف .env مع OPENAI_API_KEY")
        print("   - تثبيت جميع المتطلبات")
        print("   - تحميل البيانات في ChromaDB")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
