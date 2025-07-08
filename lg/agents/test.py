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

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# def multiply(a:float, b:float) -> float:
#     """multiply a and b

#     Args:
#     a: first float
#     b: second float

#     """
#     return a*b

# def add(a:float, b:float) -> float:
#     """add a and b

#     Args:
#     a: first float
#     b: second float

#     """
#     return a*b

# def divide(a:float, b:float) -> float:
#     """divide a and b

#     Args:
#     a: first float
#     b: second float

#     """
#     return a*b



# tools = [multiply, add, divide]

# llm_with_tools = llm.bind_tools(tools)

# sys_message = SystemMessage(content="You are a helpful assitant tasked with performing arithmatic tasks on a set of inputs")



# def agent (state: MessagesState):
#     return {'messages': [llm_with_tools.invoke([sys_message] + state['messages'])]}


# # build graph
# builder = StateGraph(MessagesState)
# builder.add_node("agent", agent)
# builder.add_node("tools", ToolNode([multiply]))

# # logic
# builder.add_edge(START, "agent")
# builder.add_conditional_edges(
#     "agent",
#     tools_condition,
# )
# builder.add_edge("tools", "agent")

# graph = builder.compile()


# # messages = [HumanMessage(content="what is 5 multiplied by 0 ", name='Joshua')]
# messages = [HumanMessage(content="what is 5 + 10 *10 +7+ 112+  7", name='Joshua')]

# messages = graph.invoke({'messages': messages}, {"recursion_limit": 100})

# for m in messages['messages']:
#     m.pretty_print()
# text = "Love"
# message = f"what is the meaning of {text}"

# response = llm.invoke(message)
# print(response)

schema = {
    "title": "MultiQueryOutput",
    "description": "Generate multiple query variations",
    "type": "object", 
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of alternative query formulations"
        }
    }
}
chat_model = ChatGroq(model="deepseek-r1-distill-llama-70b").with_structured_output(schema=schema)


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from rich import print


multi_query_prompt = ChatPromptTemplate.from_template(
    """
You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database in JSON format. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.

Original question: {question}
"""
)



# ✅ Compose a runnable chain
llm_chain = multi_query_prompt | chat_model

# ✅ Call the composed chain with a dictionary
result = llm.invoke({"question": "what is langchain?"})
print(result)

# print(result["queries"])
