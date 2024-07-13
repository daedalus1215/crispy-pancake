from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

template = "Tell me a fact about {planet}"

prompt = PromptTemplate(template=template, input_variables=["planet"])

prompt.save("myprompt.json")

from langchain.prompts import load_prompt

loaded_prompt = load_prompt("myprompt.json")
llm.invoke(loaded_prompt.format(planet='mars'))
