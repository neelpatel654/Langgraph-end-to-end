from email import message
from logging import config
from langchain_community.tools import tool
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage,RemoveMessage

memory = MemorySaver()
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

llm= ChatGoogleGenerativeAI(api_key=gemini_api_key,model="gemini-2.0-flash")

@tool
def search(query: str):
    """Call to surf the web."""
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."

tools= [search]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools=tools)

# delete message(programmatically)
# We can also delete messages programmatically from inside the graph. 
# Here we'll modify the graph to delete any old messages (longer than 3 messages ago) at the end of a graph run.
def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages":[RemoveMessage(id= m.id) for m in messages[:-3]]}

def should_continue(state:MessagesState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    else:
        return "action"

def call_model(state:MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":response}

workflow = StateGraph(MessagesState)

workflow.add_node("agent",call_model)
workflow.add_node("action",tool_node)
workflow.add_node(delete_messages)

workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent",
                               should_continue,
                               {
                                   END:END,
                                   "action":"action"
                               })
workflow.add_edge("action","agent")
workflow.add_edge("delete_messages",END)

app = workflow.compile(checkpointer=memory)

config = {"configurable":{"thread_id":2}}

input_message = HumanMessage(content="hi! my name is neal")
for event in app.stream({"messages":[input_message]},config=config,stream_mode="values"):
    event["messages"][-1].pretty_print()

input_message = HumanMessage(content="what is my name??")
for event in app.stream({"messages":[input_message]},config=config,stream_mode="values"):
    event["messages"][-1].pretty_print()

messages = app.get_state(config).values["messages"]
print(messages)

# delete message(manually)
# app.update_state(config,{"messages": RemoveMessage(id=messages[0].id)}) 

# messages = app.get_state(config).values["messages"]
# print(messages)

config = {"configurable":{"thread_id":"3"}}

input_message = HumanMessage(content="hi! i am neal")
for event in app.stream({"messages":[input_message]},config=config,stream_mode="values"):
    print([(message.type,message.content) for message in event["messages"]])

input_message=  HumanMessage(content="what is my name?")
for event in app.stream({"messages":[input_message]},config=config,stream_mode="values"):
    print([(message.type,message.content)for message in event["messages"]])
