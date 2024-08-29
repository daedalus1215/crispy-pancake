from ollama_factory import OllamaFactory
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

result = 123292 * 293423
print(f"The result is {result}")
llm = OllamaFactory().create_llm()
tools = load_tools(["llm-math"], llm=llm)

dir(AgentType)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


agent.run("Calculate 123292 times 293423")