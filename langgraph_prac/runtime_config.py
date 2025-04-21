import os
import operator
from tkinter import END
from typing import Annotated, Sequence, TypedDict
from urllib import response
from git import Optional
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.runnables.config import RunnableConfig

gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=gemini_api_key)
groq = ChatGroq(model="deepseek-r1-distill-llama-70b",api_key=groq_api_key)

models = {
    "gemini": gemini,
    "groq": groq
}
class ConfigSchema(TypedDict):
    model:Optional[str]
    system_message:Optional[str]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]

# def call_model(state):
#     state["messages"]
#     response = gemini.invoke(state["messages"])
#     return {"messages":[response]}

# def call_model(state:AgentState,config: RunnableConfig):
#     model_name = config["configurable"].get("model","gemini")
#     model = models[model_name]
#     response = model.invoke(state["messages"])
#     return {"messages":[response]}

def call_model(state:AgentState,config:RunnableConfig):
    model_name = config["configurable"].get("model","gemini")
    model = models[model_name]
    messages = state["messages"]
    if "system_message" in config["configurable"]:
        messages = [
            SystemMessage(content=config["configurable"]["system_message"])
        ] + messages
    response = model.invoke(messages)
    return {"messages":[response]}

builder = StateGraph(AgentState,ConfigSchema)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

graph = builder.compile()

print(graph.invoke({"messages":[HumanMessage(content="hi")]}))