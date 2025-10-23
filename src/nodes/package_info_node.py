# src/nodes/package_info_node.py

from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from utils.retrievers import RetrieverManager
from typing import Optional, Type
from pydantic import BaseModel, Field

class PackageInfoInput(BaseModel):
    """Input for package info tool."""
    package_query: str = Field(description="استفسار المستخدم عن باقة معينة")

class PackageInfoTool(BaseTool):
    """Tool for providing details about specific packages."""
    name: str = "package_info_tool"
    description: str = """
استخدم هذه الأداة للحصول على تفاصيل باقة محددة بالاسم.

متى تستخدم هذه الأداة:
✅ "عايز تفاصيل فليكس ٧٠"
✅ "معلومات عن Plus 155"
✅ "ايه مميزات باقة ٧٠؟"
✅ "تفاصيل الباقة السابقة" (إذا ذُكرت في المحادثة)

متى لا تستخدمها:
❌ "ايه كل الباقات؟" → استخدم package_recommendation_tool
❌ "رشحلي باقة" → استخدم package_recommendation_tool
❌ "ايه أرخص باقة؟" → استخدم package_recommendation_tool

المدخل المطلوب: اسم الباقة بالضبط أو رقمها (مثل: فليكس ٧٠، باقة ٧٠، Plus 155)
"""
    args_schema: Type[BaseModel] = PackageInfoInput

    def __init__(self, retriever_manager: RetrieverManager, memory: ConversationBufferMemory):
        super().__init__()
        self._retriever_manager = retriever_manager
        self._memory = memory

    def _run(self, package_query: str, session_id: Optional[str] = None) -> str:
        """Run the package info tool."""
        docs = self._retriever_manager.get_documents(package_query, retriever_type="package")
        if not docs:
            return "عذراً، لم أجد باقة بهذا الاسم في قاعدة البيانات. يرجى التأكد من اسم الباقة أو تجربة باقة أخرى."
        else:
            # Found documents - extract information directly without LLM processing
            context = docs[0].page_content
            
            # Parse the content to extract structured information
            lines = context.split(' — ')
            if len(lines) >= 3:
                title = lines[0].strip()
                details = lines[1].strip()
                price = lines[2].strip()
                
                # Format the response directly from the data
                response = f"اسم الباقة: {title}\nالتفاصيل: {details}\nالسعر: {price}"
            else:
                # Check if this is a search variation - get original content from metadata
                if hasattr(docs[0], 'metadata') and 'original_content' in docs[0].metadata:
                    original_content = docs[0].metadata['original_content']
                    lines = original_content.split(' — ')
                    if len(lines) >= 3:
                        title = lines[0].strip()
                        details = lines[1].strip()
                        price = lines[2].strip()
                        response = f"اسم الباقة: {title}\nالتفاصيل: {details}\nالسعر: {price}"
                    else:
                        response = f"هذه المعلومات المتاحة عن الباقة:\n{original_content}"
                else:
                    # Fallback to showing raw content
                    response = f"هذه المعلومات المتاحة عن الباقة:\n{context}"

        return response

    async def _arun(self, package_query: str, session_id: Optional[str] = None) -> str:
        """Async run method."""
        return self._run(package_query, session_id)

