from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.schema import HumanMessage,SystemMessage # type: ignore
from langchain_community.cache import InMemoryCache # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

llm.cache = InMemoryCache()

## Use templates
# single_input = PromptTemplate(input_variables=["topic"], 
#                               template='Tell me a fact about {topic}')

# llm.invoke(single_input.format(topic="Mars"))

multi_input_prompt = PromptTemplate(input_variables=['topic', 'level'],
                                    template='Tell me a fact about {topic} for a {level}')

llm(multi_input_prompt.format(topic='the ocean', level='PhD level'));