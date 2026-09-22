import json
import os
from typing import Optional
from crewai.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class SupportRAGTool(BaseTool):
    name: str = "Customer Knowledge Base Search"
    description: str = (
        "Search customer orders and profile details using query terms or IDs. "
        "Returns details including Customer ID, order date, contact number, status, and shopping address."
    )
    vector_store: Optional[FAISS] = None

    def __init__(self, json_path: str = "knowledge_base.json", openai_api_key: Optional[str] = None):
        super().__init__()
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Knowledge base file not found at {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Chunk the data into text documents
        docs = []
        for record in data:
            content = (
                f"Customer ID: {record.get('customer_id')}\n"
                f"Customer Name: {record.get('customer_name')}\n"
                f"Contact Number: {record.get('contact_number')}\n"
                f"Order ID: {record.get('order_id')}\n"
                f"Order Date: {record.get('order_date')}\n"
                f"Status: {record.get('status')}\n"
                f"Shopping Address: {record.get('shopping_address')}\n"
                f"Items Ordered: {', '.join(record.get('items', []))}\n"
                f"Notes: {record.get('notes')}"
            )
            docs.append(Document(page_content=content, metadata={"customer_id": record.get("customer_id")}))

        # Create embeddings and build FAISS vector database
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        object.__setattr__(self, "vector_store", FAISS.from_documents(docs, embeddings))

    def _run(self, query: str) -> str:
        if not self.vector_store:
            return "Vector store is not initialized."
        results = self.vector_store.similarity_search(query, k=2)
        if not results:
            return "No relevant customer record found."
        return "\n---\n".join([doc.page_content for doc in results])
