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

def multiply(a:float, b:float) -> float:
    """multiply a and b

    Args:
    a: first float
    b: second float

    """
    return a*b

def add(a:float, b:float) -> float:
    """add a and b

    Args:
    a: first float
    b: second float

    """
    return a*b

def divide(a:float, b:float) -> float:
    """divide a and b

    Args:
    a: first float
    b: second float

    """
    return a*b



tools = [multiply, add, divide]

llm_with_tools = llm.bind_tools(tools)

sys_message = SystemMessage(content="You are a helpful assitant tasked with performing arithmatic tasks on a set of inputs")



def agent (state: MessagesState):
    return {'messages': [llm_with_tools.invoke([sys_message] + state['messages'])]}


# build graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([multiply]))

# logic
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
)
builder.add_edge("tools", "agent")

graph = builder.compile()


# messages = [HumanMessage(content="what is 5 multiplied by 0 ", name='Joshua')]
messages = [HumanMessage(content="what is 5 + 10 *10 +7+ 112+  7", name='Joshua')]

messages = graph.invoke({'messages': messages}, {"recursion_limit": 100})

for m in messages['messages']:
    m.pretty_print()

