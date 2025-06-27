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

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)



def _set_env(var:str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}:")

_set_env("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] ="true"
os.environ["LANGCHAIN_PROJECT"] ="langchain-academy"



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



from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

messages = [HumanMessage(content="Multiply that by 3")]
config = {'configurable': {"thread_id": "1"}}
messages = react_graph_memory.invoke({"messages": messages}, config) ## this output will be saved and passed in as the next paremater


messages = [HumanMessage(content="Add 3 to it")]
messages = react_graph_memory.invoke({"messages": messages}, config) 

"""
basically memery is stored in the tread id, so it is call the state will be saves into the config variable, and when
passed as a paremeter to the second message the model will remeber 

checkpointer writes the start of the graph at every step and allows us to collect ss in a thread and pass into graph other invocation and pickup where it leftof 
so have access to all the states of the past


for lang studio we do not have to do the memory thing, its alr packaged with memory thing 
"""





