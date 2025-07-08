import asyncio
import os
from rich import print

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import chromadb


if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = ""

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()


embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



file_path = r"C:\Users\user\Downloads\Tan-Yu-Hang-Resume-20250623.pdf"
loader = PyPDFLoader(file_path)
pages = []

async def load_pages():
    async for page in loader.alazy_load():
        pages.append(page)

    return pages

# Run the async function using asyncio
pages = asyncio.run(load_pages())

character_splitter = CharacterTextSplitter(
    chunk_size=200,  # Size in characters
    chunk_overlap=10,  # Overlap between chunks
    separator="",
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Size in characters
    chunk_overlap=10,  # Overlap between chunks
)

split_docs = text_splitter.split_documents(pages)



client = chromadb.PersistentClient(path="./resume")
# collection = client.create_collection(name="resume")

vector_store = Chroma.from_documents(
    documents=split_docs,
    collection_name="resume",
    embedding=embedding_model,
    # persist_directory="./chroma_db"

)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# result = vector_retriever.invoke("Project management?")

# print(result)
from langchain_core.prompts import PromptTemplate


prompt = PromptTemplate.from_template(
    "You are an expert resume analyst. Carefully review the following resume and identify its strongest points. "
    "Focus on relevant skills, achievements, experience, and qualities that make the candidate stand out. "
    "Present your analysis in a concise, bullet-point format:\n\nResume:\n{pages}"
)
# response = llm.invoke(f"You are an expert in the field of AI and you are given a question and a document. You need to answer the question based on the document. Here is the question: Why does self-attention have an advantage over recurrent layers in terms of parallelization and path length for long-range dependencies? Here is the document: {result}")
chain = prompt | llm


# Convert list of Document to one string
all_text = "\n\n".join([doc.page_content for doc in pages])

# Invoke with correct variable name and format
result = chain.invoke({"pages": all_text})

print(result.content)


# prompt = "you are a resume analyser, analyse the strong points of this resume: {}"



# # Access and print the first page's metadata and content
# # print(f"{pages[0].metadata}\n")
# # print(pages[0].page_content)
# # print(pages)



# character_splitter = CharacterTextSplitter(
#     chunk_size=200,  # Size in characters
#     chunk_overlap=10,  # Overlap between chunks
#     separator="",
# )

# # chunks = character_splitter.split_text(pages)

# # print(f"Total number of chunks: {len(chunks)}")

# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i+1}:")
# #     print(f"Length: {len(chunk)} characters")
# #     print(f"Content: {chunk}")
# #     print("-"*30)

# # ✅ Use this method to split Document objects
# # split_docs = character_splitter.split_documents(pages)

# # print(f"Total number of chunks: {len(split_docs)}")

# # for i, doc in enumerate(split_docs):
# #     print(f"Chunk {i+1}:")
# #     print(f"Length: {len(doc.page_content)} characters")
# #     print(f"Content: {doc.page_content}")
# #     print("-" * 30)




# print(f"Total number of chunks: {len(split_docs)}")

# for i, doc in enumerate(split_docs):
#     print(f"Chunk {i+1}:")
#     print(f"Length: {len(doc.page_content)} characters")
#     print(f"Content: {doc.page_content}")
#     print("-" * 30)












