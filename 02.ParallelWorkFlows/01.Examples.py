# we will get the data of the batsman then we will calulate the SR, boundary /ball
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




class BatsmanState(TypedDict):
  name:str
  runs:int
  balls:int

  fours:int
  sixes:int
  sr:float
  bpb:float
  boundary_percentage:float



def calculate_sr(state:BatsmanState)->BatsmanState:
  sr = state['runs']/state['balls']*100
  return {'sr':sr}


def calculate_bpb(state:BatsmanState)->BatsmanState:
  bpb = (state['fours']+state['sixes'])/state['balls']
  return {'bpb':bpb}

def calculate_boundary_percentage(state:BatsmanState)->BatsmanState:
  boundary_percentage = (state['fours']+state['sixes'])/state['runs']
  return {'boundary_percentage':boundary_percentage}

def summary(state:BatsmanState)->BatsmanState:
  s = '''
     Name: {state["name"]}
     Runs: {state["runs"]}
     Balls: {state["balls"]}
     Fours: {state["fours"]}
     Sixes: {state["sixes"]}
     SR: {state["sr"]}
     BPB: {state["bpb"]}
     Boundary Percentage: {state["boundary_percentage"]}
  '''
  summary = s

  return {'summary':summary}


graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr',calculate_sr)
graph.add_node('calculate_bpb',calculate_bpb)
graph.add_node('calculate_boundary_percentage',calculate_boundary_percentage)
graph.add_node('summary',summary)

graph.add_edge(START,'calculate_sr')
graph.add_edge(START,'calculate_bpb')
graph.add_edge(START,'calculate_boundary_percentage')
graph.add_edge('calculate_sr','summary')
graph.add_edge('calculate_bpb','summary')
graph.add_edge('calculate_boundary_percentage','summary')
graph.add_edge('summary',END)


workflow = graph.compile()


init_state={
    'name':"pradeep kumar",
    'runs':100,
    'balls':30,
    'fours':3,
    'sixes':13
}

final_state = workflow.invoke(init_state)
print(final_state)