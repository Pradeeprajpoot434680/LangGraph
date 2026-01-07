from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os

load_dotenv()


llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        task='text-generation',
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

h_model = ChatHuggingFace(llm=llm)
g_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')


#create state

class LLMState(TypedDict):
  question:str
  answer:str


def llm_qa(state:LLMState)->LLMState:
  # extract question
  question = state['question']
  #form prompt
  prompt = f'Answer the following question: {question}'
  # invoke model
  answer = h_model.invoke(prompt).content
  #update
  state['answer'] =answer
  return state



#graph

graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa',llm_qa)

# edges
graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

# compile




init_state={'question':'how far is moon from the earth?'}
final_state = workflow.invoke(init_state)

print(final_state)