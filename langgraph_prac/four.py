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
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
load_dotenv()
memory = MemorySaver()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)

tavily_tool = TavilySearchResults(max_results=2)

@tool
def human_assistant(query: str)->str:
    """
    Request help from a human assistant when the AI is unsure or external expertise is required.
    """
    human_response = interrupt({"query":query})
    return human_response["data"]

tools = [tavily_tool,human_assistant]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)

class State(TypedDict):
    messages : Annotated[list,add_messages]
    name: str
    birthday: str

def chatbot(state : State):
    message = llm_with_tools.invoke( state["messages"])
    # print("LLM Message:", message)
    # print("Tool Calls:", message.tool_calls)
    return {"messages":[message]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_conditional_edges("chatbot",
                            tools_condition,)

graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable":{"thread_id":"1"}}

from IPython.display import Image,display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# user_input = "I need some expert guidance for building an AI agent.give me basic concepts of building an AI agent If you can't answer, please ask a human assistant using the appropriate tool."
user_input = "retrieve the age of narendra modi from any search tool and then please ask age of human by human assistant using the appropriate tool and compare both age."
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)

print(snapshot.next)
human_response = (
    "my age is 21 years."
)
human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

def stream_graph_update(user_input: str):
    for event in graph.stream({"messages": [{"role":"user","content":user_input}]},config=config,stream_mode="values"):
            event["messages"][-1].pretty_print()


# while snapshot.next == ('tools',):
#     human_response = input("🔍 Human assistant needed. Please provide your input: ")

#     human_command = Command(resume={"data": human_response})

#     events = graph.stream(human_command, config, stream_mode="values")
#     for event in events:
#         if "messages" in event:
#             event["messages"][-1].pretty_print()
#     snapshot = graph.get_state(config)

# events = graph.stream(
#     {"messages":[{"role":"user","content":user_input}]},
#     config=config,
#     stream_mode="values"
# )

# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()


# events = graph.stream(human_command, config, stream_mode="values")
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

# def stream_graph_update(user_input: str):
#     for event in graph.stream({"messages": [{"role":"user","content":user_input}]},config=config,stream_mode="values"):
#             event["messages"][-1].pretty_print()

# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["q","quit","exit"]:
#             print("goodbye")
#             break
#         stream_graph_update(user_input)
#     except:
#         print("please pass input")
