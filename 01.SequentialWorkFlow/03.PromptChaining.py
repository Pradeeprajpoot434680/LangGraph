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


# generate a blog
# start -> generate outline -> gen blog -> end

class BlogState(TypedDict):
  title:str
  outline:str
  content:str

def generate_outline(state:BlogState)->BlogState:
  # extract title
  title = state['title']
  #form prompt
  prompt = f'Generate a outline for a blog on the topic: {title}'
  # invoke model
  outline = h_model.invoke(prompt).content
  #update
  state['outline'] =outline
  return state


def generate_blog(state:BlogState)->BlogState:
  # extract title
  title = state['title']
  outline = state['outline']
  #form prompt
  prompt = f'Generate a Blog in the basis of title and outline. title is -> {title} and outline is {outline}'
  # invoke model
  blog = h_model.invoke(prompt).content
  #update
  state['content'] =blog
  return state

graph = StateGraph(BlogState)

graph.add_node('generate_outline',generate_outline)
graph.add_node('generate_blog',generate_blog)

graph.add_edge(START,'generate_outline')
graph.add_edge('generate_outline','generate_blog')
graph.add_edge('generate_blog',END)
workflow = graph.compile()


init_state={'title':'Rise of AI in India'}
final_state = workflow.invoke(init_state)
print(final_state)