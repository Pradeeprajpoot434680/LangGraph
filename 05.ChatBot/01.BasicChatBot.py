# we will get the data of the batsman then we will calulate the SR, boundary /ball
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage
from pydantic import BaseModel, Field
import operator
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        task='text-generation',
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

h_model = ChatHuggingFace(llm=llm)

# Use gemini-2.5-flash instead of the retired 1.5 version
g_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key="AIzaSyDp7Op3hfgHIKxfxmWzrB4Js70mTsBl77k"
)


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# ... (imports remain the same)

checkpointer = MemorySaver()
graph = StateGraph(ChatState)

def chat_node(state: ChatState):
    # This now works correctly because the checkpointer is active
    messages = state['messages']
    res = h_model.invoke(messages)
    return {'messages': [res]}

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# IMPORTANT: Added checkpointer here
workflow = graph.compile(checkpointer=checkpointer)

thread_id = "1" # Thread IDs should ideally be strings
config = {'configurable': {'thread_id': thread_id}}

while True:
    user_input = input("Type Here...")
    if user_input.strip().lower() in ['exit', 'quit', 'bye']:
        break

    # invoke with the config so the graph knows which thread to look up
    res = workflow.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    
    # LangGraph add_messages will have appended the new AI message to the end of the list
    print("AI:", res['messages'][-1].content)


# to see all the messages of a perticular id

workflow.get_state(config=config)