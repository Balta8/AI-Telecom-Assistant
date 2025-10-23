# src/retrievers.py
from typing import List
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, EMBEDDING_MODEL, require_openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

class RetrieverManager:
    def __init__(self, persist_directory: str, embedding_model: str = EMBEDDING_MODEL, k: int = 20):
        self.embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=embedding_model)
        self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_model)
        self.retrievers = {}
        self.k = k

    @staticmethod
    def normalize_numbers(text: str) -> str:
        arabic_nums = "٠١٢٣٤٥٦٧٨٩"
        english_nums = "0123456789"
        table = str.maketrans(arabic_nums, english_nums)
        return text.translate(table)

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """تستخرج الأرقام من النص"""
        return re.findall(r"\d+", RetrieverManager.normalize_numbers(text))

    @staticmethod
    def clean_docs(docs: List[Document]) -> List[Document]:
        """فلترة docs اللي فيها nan أو فاضية"""
        cleaned_docs = []
        for doc in docs:
            if doc.page_content and doc.page_content.strip() != "":
                # Remove trailing "nan" from content
                content = doc.page_content.strip()
                if content.endswith(" — nan"):
                    content = content[:-6]  # Remove " — nan"
                elif content.endswith("nan"):
                    content = content[:-3]  # Remove "nan"
                
                if content and content.strip() != "":
                    # Create new document with cleaned content
                    cleaned_doc = Document(
                        page_content=content.strip(),
                        metadata=doc.metadata
                    )
                    cleaned_docs.append(cleaned_doc)
        return cleaned_docs

    def setup_retrievers(self):
        self.retrievers['faq'] = self.db.as_retriever(search_kwargs={"k": 4, "filter": {"type": "faq"}})
        self.retrievers['package'] = self.db.as_retriever(search_kwargs={"k": 6, "filter": {"type": "package"}})

    def rerank_with_llm(self, query: str, docs: List[Document]) -> List[Document]:
        """LLM re-ranking للـ docs"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=require_openai_key())

        prompt = PromptTemplate(
            input_variables=["query", "docs"],
            template="""السؤال: {query}

الوثائق:
{docs}

اختار أفضل الوثائق (بالترتيب) اللي بتجاوب على السؤال. ارجع النصوص فقط بنفس الترتيب."""
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        docs_text = "\n\n".join([f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)])
        result = chain.run(query=query, docs=docs_text)

        ranked_docs = []
        for doc in docs:
            if doc.page_content.strip() and doc.page_content in result:
                ranked_docs.append(doc)

        return ranked_docs if ranked_docs else docs

    def get_documents(self, query: str, retriever_type: str) -> List[Document]:
        retriever = self.retrievers.get(retriever_type)
        if not retriever:
            raise ValueError(f"Retriever '{retriever_type}' غير موجود")

        # لو Package → expand query قبل البحث
        if retriever_type == "package":
            query = self._expand_package_query(query)

        docs = retriever.get_relevant_documents(query)

        # تنظيف النتائج (حذف nan أو الفاضية)
        docs = self.clean_docs(docs)

        # لو FAQ → semantic فقط
        if retriever_type == "faq":
            return docs

        # لو Package → improve search
        if retriever_type == "package":
            docs = self._improve_package_search(query, docs)
            
            # فلترة أرقام إذا وُجدت
            query_numbers = self.extract_numbers(query)
            if query_numbers:
                filtered_docs = [
                    doc for doc in docs
                    if any(num in self.normalize_numbers(doc.page_content) for num in query_numbers)
                ]
                if filtered_docs:
                    docs = filtered_docs
                else:
                    # مفيش أي doc يطابق الرقم المطلوب → رجّع فاضي
                    return []

        # rerank بالـ LLM
        return self.rerank_with_llm(query, docs)
    
    def _expand_package_query(self, query: str) -> str:
        """توسيع الاستعلام لتحسين البحث
        
        مثال: "باقة ٧٠" → "فليكس ٧٠"
        """
        query_lower = query.lower()
        
        # إذا كان الاستعلام يحتوي على "باقة" أو "باقه" بدون "فليكس" أو "plus"
        if ("باقة" in query or "باقه" in query) and "فليكس" not in query and "plus" not in query_lower:
            # استخرج الأرقام من الاستعلام
            numbers = self.extract_numbers(query)
            if numbers:
                # أضف "فليكس" للاستعلام لتحسين البحث
                query = f"فليكس {numbers[0]}"
        
        return query
    
    def _improve_package_search(self, query: str, docs: List[Document]) -> List[Document]:
        """تحسين البحث للباقات باستخدام فهم أفضل للسياق"""
        if not docs:
            return docs
            
        # تحليل الاستعلام لفهم النية
        query_lower = query.lower()
        query_normalized = self.normalize_numbers(query)
        
        # إذا كان الاستعلام يحتوي على "فليكس" أو "باقة" (نعتبرهم نفس الشيء)
        # مثال: "باقة ٧٠" = "فليكس ٧٠"
        if "فليكس" in query or "flex" in query_lower or "باقة" in query or "باقه" in query:
            # ابحث عن باقات فليكس أولاً
            flex_docs = [doc for doc in docs if "فليكس" in doc.page_content]
            if flex_docs:
                return flex_docs
            # إذا لم نجد فليكس، ارجع كل النتائج (ممكن يكون Plus أو غيرها)
        
        # إذا كان الاستعلام يحتوي على "plus" أو "بلس"
        if "plus" in query_lower or "بلس" in query:
            plus_docs = [doc for doc in docs if "Plus" in doc.page_content or "plus" in doc.page_content.lower()]
            if plus_docs:
                return plus_docs
        
        # إذا لم نجد تطابق محدد، نرجع النتائج الأصلية
        return docs
