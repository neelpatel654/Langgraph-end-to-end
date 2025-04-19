from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool,InjectedToolCallId
from langchain_core.messages import HumanMessage,ToolMessage

load_dotenv()
memory = MemorySaver()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)

class State(TypedDict):
    messages : Annotated[list,add_messages]
    name: str
    birthday: str

# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
@tool
def human_assistance(name:str,birthday:str,tool_call_id:Annotated[str,InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
    {
        "question":"Is this Correct?",
        "name":name,
        "birthday":birthday,
    },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct","").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name",name)
        verified_birthday = human_response.get("birthday",birthday)
        response = f"Made a correction: {human_response}"
    
    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update ={
        "name":verified_name,
        "birthday":verified_birthday,
        "messages":[ToolMessage(response,tool_call_id=tool_call_id)]
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)

tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool,human_assistance]

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state:State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages":[message]}

graph_builder = StateGraph(State)

tool_node= ToolNode(tools=tools)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges("chatbot",tools_condition)
graph_builder.add_edge("tools","chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)

config = {"configurable":{"thread_id":"1"}}

events = graph.stream(
    {"messages":[{"role":"user","content":user_input}]},
    config=config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

human_command = Command(
    resume={
        "name":"LangGraph",
        "birthday":"Jan 17, 2024",
    },
)

events = graph.stream(human_command,config,stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()