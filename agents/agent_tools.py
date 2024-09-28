from langchain.agents import initialize_agent
from langchain.agents import AgentType
from ollama_factory import OllamaFactory
from langchain_community.agent_toolkits.load_tools import load_tools
import os


os.environ['SERPAPI_API_KEY'] = "411c17eae6b0b007d12f2a303731c6074e731e5c438f1279ecdb22a33f8d797d"


llm =  OllamaFactory().create_llm()
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What year was ALbert Einstein born? What is that year multiplied by 5?")
