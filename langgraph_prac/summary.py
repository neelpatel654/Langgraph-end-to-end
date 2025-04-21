from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langgraph.graph.message import MessagesState
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

memory = MemorySaver()
gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)

class State(MessagesState):
    summary: str

def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get("summary","")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages":[response]}    

# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state:State) -> Literal["summarize_conversation",END] :
    """Return the next node to execute"""
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    else:
        return END

def summarize_conversation(state: State):
    summary = state.get("summary","")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary":response.content,"messages":delete_messages}

workflow = StateGraph(State)

workflow.add_node("conversation",call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START,"conversation")
workflow.add_conditional_edges("conversation",should_continue)
workflow.add_edge("summarize_conversation",END)

app = workflow.compile(checkpointer=memory)

def print_update(update):
    for k,v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])

config = {"configurable":{"thread_id":"4"}}
input_message= HumanMessage(content="Hello,my name is neal?")
input_message.pretty_print()
for event in app.stream({"messages":[input_message]},config,stream_mode="updates"):
    print_update(event)

input_message= HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages":[input_message]},config,stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the Gujarat Titans!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

values = app.get_state(config).values
print(values)

input_message = HumanMessage(content="i like how much they win")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

values = app.get_state(config).values
print(values)