from ollama_factory import OllamaFactory
from langchain.agents import create_react_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.prompts import PromptTemplate
from langchain.schema import PromptValue
from typing import List

# Create a custom prompt template by subclassing PromptTemplate
class CustomPromptTemplate(PromptTemplate):
    def __init__(self, template: str):
        # Ensure correct initialization of the base class
        super().__init__()
        self.template = template
        self._input_variables = ["tools", "tool_names", "agent_scratchpad"]

    def format(self, **kwargs) -> str:
        # Format the template with the provided arguments
        return self.template.format(**kwargs)

    def format_prompt(self, **kwargs) -> PromptValue:
        # This should return a PromptValue object
        prompt_text = self.format(**kwargs)
        return PromptValue(text=prompt_text)

    def input_variables(self) -> List[str]:
        # Return the variables that the template expects
        return self._input_variables

# Direct multiplication for comparison
result = 123292 * 293423
print(f"The direct result is: {result}")

# Create the LLM instance
llm = OllamaFactory().create_llm()
llm.temperature = 0

# Load tools
try:
    tools = load_tools(["llm-math"], llm=llm)
except Exception as e:
    print(f"Error loading tools: {e}")
    tools = []

# Initialize the agent using the custom prompt template
try:
    tool_names = ", ".join([tool.name for tool in tools])  # List of tool names
    prompt_template = CustomPromptTemplate(template="""
    You are a helpful assistant with the following tools: {tool_names}.
    You should use the tools to assist with any tasks given to you.

    Begin!

    {agent_scratchpad}
    """)
    
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt_template=prompt_template,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
except Exception as e:
    print(f"Error initializing agent: {e}")
    agent = None

# Run the agent using the updated invoke method
if agent:
    try:
        results = agent.invoke("Calculate 123292 times 293423")
        print(f"Agent result: {results}")
    except Exception as e:
        print(f"Error running agent: {e}")
