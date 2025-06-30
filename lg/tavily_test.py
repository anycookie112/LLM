from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()



from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph, START, END
import getpass
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from pprint import pprint

import datetime
from datetime import datetime, timedelta

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

"""
tool call section
"""

### Build Index

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List


# from langchain_openai import OpenAIEmbeddings

### from langchain_cohere import CohereEmbeddings

# Set embeddings
# embd = OpenAIEmbeddings()

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)



# print(doc_splits)


class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()


embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# # Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedding_model,
)
retriever = vectorstore.as_retriever()

### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# so this is to test, where the llm will get the data from, so the first question when unrelated to whats in the vector database it will go to the web
question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))








# # retrieved_docs = vectorstore.similarity_search("what are agents?", k=3)
# # for doc in retrieved_docs:
# #     print(doc.page_content)

# query = "how do agents work?"
# retrieved_docs = retriever.get_relevant_documents(query)

# context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# prompt = f"""
# Answer the question based only on the context below and also give citations based on the article you referenced.

# Context:
# {context}

# Question: {query}
# Answer:
# """

# response = llm.invoke(prompt)
# print(response.content)


# # print(retriever)







# tavily_client = TavilyClient(api_key="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")
# response = tavily_client.search("Who is Leo Messi?")
# print(response)