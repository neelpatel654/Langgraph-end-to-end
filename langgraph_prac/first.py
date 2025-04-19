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

load_dotenv()
memory = MemorySaver()

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)

tool = TavilySearchResults(max_results=3)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=[tool])

class State(TypedDict):
    messages : Annotated[list,add_messages]

def chatbot(state : State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}


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

# user_input = "hello, what is my name?."
# events = graph.stream({"messages":[{"role":"user","content":user_input}]},
#                                     config=config,
#                                     stream_mode="values") 
# for event in events:
#     event["messages"][-1].pretty_print()         

def stream_graph_update(user_input: str):
    for event in graph.stream({"messages": [{"role":"user","content":user_input}]},config=config,stream_mode="values"):
            event["messages"][-1].pretty_print()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["q","quit","exit"]:
            print("goodbye")
            break
        stream_graph_update(user_input)
    except:
        print("please pass input")
