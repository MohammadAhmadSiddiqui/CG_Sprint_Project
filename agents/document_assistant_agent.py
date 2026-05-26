import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from typing import List

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14")
CHAT_DEPLOY    = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
EMBED_DEPLOY   = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

KNOWLEDGE_DIR    = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")


# ── Custom embeddings class that calls Foundry endpoint directly ──────────────
class FoundryEmbeddings(Embeddings):
    def _embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{AZURE_ENDPOINT}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_API_KEY,
        }
        payload = {
            "model": EMBED_DEPLOY,
            "input": texts,
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


def build_vector_store():
    texts = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.create_documents(texts)
    vector_store = FAISS.from_documents(chunks, FoundryEmbeddings())
    vector_store.save_local(VECTOR_STORE_DIR)
    print("Vector store built and saved.")
    return vector_store


def load_vector_store():
    if os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
        return FAISS.load_local(VECTOR_STORE_DIR, FoundryEmbeddings(), allow_dangerous_deserialization=True)
    return build_vector_store()


class DocumentAssistantAgent:
    def __init__(self):
        self.client       = OpenAI(api_key=AZURE_API_KEY, base_url=AZURE_ENDPOINT)
        self.vector_store = load_vector_store()
        self.retriever    = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def run(self, question: str) -> str:
        docs    = self.retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""You are a helpful Medical Document Assistant specialized in diabetes.
Use ONLY the context below to answer. If the answer is not in the context, say
"I don't have enough information on that."

Context:
{context}

Question: {question}

Answer (clear and concise):"""

        response = self.client.chat.completions.create(
            model=CHAT_DEPLOY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()