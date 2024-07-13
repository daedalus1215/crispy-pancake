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

multi_input_prompt = PromptTemplate(input_variables=['topic', 'level'],
                                    template='Tell me a fact about {topic} for a {level}')

llm.invoke(multi_input_prompt.format(topic='the ocean', level='3rd grade'))