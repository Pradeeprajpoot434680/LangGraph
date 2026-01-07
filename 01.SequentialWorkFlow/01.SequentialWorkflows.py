from langgraph.graph import StateGraph,START,END
from typing import TypedDict

# bmi states
class BMIState(TypedDict):
  weight_kg:float
  height_m:float
  bmi:float
  category:str


def calculate_bmi(state:BMIState)->BMIState:
    state['bmi'] = state['weight_kg']/(state['height_m']**2)
    return state


def label_bmi(state:BMIState)->BMIState:
  if state['bmi']<18.5:
    state['category'] = 'Underweight'
  elif state['bmi']<25:
    state['category'] = 'Normal'
  elif state['bmi']<30:
    state['category'] = 'Overweight'
  else:
    state['category'] = 'Obese'

  return state


# define your graph
graph = StateGraph(BMIState)

# add nodes
graph.add_node('calculate_bmi',calculate_bmi)
graph.add_node('label_bmi',label_bmi)

# add edges
graph.add_edge(START,'calculate_bmi')
graph.add_edge('calculate_bmi','label_bmi')
graph.add_edge('label_bmi',END)


# compile the graph

workflow = graph.compile()

#execute the graph

init_state = { 'weight_kg':80,'height_m':1.73}
final_state = workflow.invoke(init_state)

print(final_state)

from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())