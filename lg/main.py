
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


def _set_env(var:str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}:")

_set_env("GROQ_API_KEY")


"""
chaining messages section
"""

# messages = [AIMessage(content=f"So you said you were researching weird ice cream flavours?", name="Model")]
# messages.extend([HumanMessage(content=f"Yes, thats right.", name="Joshua")])
# messages.extend([AIMessage(content=f"Great what would you like to learn about", name= "Model")])
# messages.extend([HumanMessage(content=f"I would like to learn about special ice creams all over the world.", name="Joshua")])

# for m in messages:
#     m.pretty_print()
    




# result = llm.invoke(messages)
# type(result)

# print(result)


# print(result.response_metadata)


"""
tool call section
"""

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

def multiply(a:int, b:int) -> int:
    """multiply a and b

    Args:
    a: first int
    b: second int

    """
    return a*b



llm_with_tools = llm.bind_tools([multiply])



def tool_calling_llm (state: MessagesState):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}


# build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

# logic
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", END)

graph = builder.compile()


# messages = [HumanMessage(content="what is 5 multiplied by 0 ", name='Joshua')]
messages = [HumanMessage(content="what is 5 + 10", name='Joshua')]

messages = graph.invoke({'messages': messages})

for m in messages['messages']:
    m.pretty_print()



# tool_call = llm_with_tools.invoke([HumanMessage(content=f"what is 5 multiplied by 0", name='Joshua')])
# print(tool_call)
# print(tool_call.additional_kwargs['tool_calls'])


"""

appending messages into message state for memory 

"""

# class MessagesState (TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]


# class State(MessagesState):
    
#     pass



# initial_messages = [AIMessage(content="Hello, how can i assist you?", name='Model'),
#                     HumanMessage(content="I am looking for information about love.", name='Joshua')]

# new_messages = AIMessage(content="Sure, I can help you with that, what are you specifically interested in?", name='Model')

# combined = add_messages(initial_messages, new_messages)
# print(combined)