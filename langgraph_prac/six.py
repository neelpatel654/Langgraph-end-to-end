from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import Annotated
from sympy import content
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)

tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]

llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable":{"thread_id":"1"}}

events = graph.stream(
    {
    "messages": [{"role":"user","content":(
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),}]
    },
    config=config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages":[{"role":"user","content":(
            "Ya that's helpful. Maybe I'll "
            "build an autonomous agent with it!"
        )}]
    },
    config=config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

to_replay = None
for state in graph.get_state_history(config=config):
    print("Num Message: ",len(state.values["messages"]), "Next: ",state.next)
    print("-"*80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state

for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()