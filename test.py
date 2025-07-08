# import gradio as gr
# import os 
# from langchain_groq import ChatGroq

# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = ""

# llm = ChatGroq(
#     model="deepseek-r1-distill-llama-70b",
#     temperature=0,
#     max_tokens=None,
#     reasoning_format="parsed",
#     timeout=None,
#     max_retries=2,
#     # other params...
# )



# def chat(user_input, history=[]):
#     response = llm(user_input)  # your LLM call
#     history.append((user_input, response))
#     return history, history

# gr.ChatInterface(fn=chat).launch()

from langchain_community.document_loaders import PyPDFLoader
from rich import print
import asyncio
from typing import List


import os
from langchain.document_loaders import PyPDFLoader

folder_path = r"C:\Users\user\Desktop\resume"
pages = []

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()


embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# async def load_documents(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(folder_path, filename)
#             loader = PyPDFLoader(file_path)
#             async for page in loader.alazy_load():
#                 pages.append(page)
#     return pages

async def load_documents(folder_path):
    pages = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)

            # Load all pages
            file_pages = []
            async for page in loader.alazy_load():
                file_pages.append(page)

            if file_pages:
                # Try to extract name from the first page's content
                first_lines = file_pages[0].page_content.strip().splitlines()
                name = None
                for line in first_lines:
                    clean_line = line.strip()
                    if clean_line and "@" not in clean_line and not clean_line.lower().startswith("summary"):
                        name = clean_line
                        break
                name = name or "Unknown"

                # Add name to each page's metadata
                for page in file_pages:
                    page.metadata["name"] = name
                    pages.append(page)
    return pages


pages = asyncio.run(load_documents(folder_path))

print(pages)


# from langchain_core.vectorstores import InMemoryVectorStore
# # from langchain_openai import OpenAIEmbeddings

# vector_store = InMemoryVectorStore.from_documents(pages, embedding_model)
# docs_with_scores = vector_store.similarity_search_with_score("React", k=5)

# # print(docs_with_scores)
# for doc, score in docs_with_scores:
#     print(f"\n--- Match (Score: {score:.4f}) ---")
#     print(f"File: {doc.metadata.get('source')}")
#     print(f"Page: {doc.metadata.get('page')} | Label: {doc.metadata.get('page_label')}")
#     print(f"Snippet: {doc.page_content[:300]}...")




# docs = vector_store.similarity_search("React", k=2)
# import hashlib

# def hash_text(text):
#     return hashlib.md5(text.encode('utf-8')).hexdigest()

# seen_hashes = set()
# unique_docs = []

# for doc in docs:
#     h = hash_text(doc.page_content)
#     if h not in seen_hashes:
#         unique_docs.append(doc)
#         seen_hashes.add(h)



# so for this part i would need to only show 2-3 candidates with no duplicates






