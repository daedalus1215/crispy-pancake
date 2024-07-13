from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.cache import InMemoryCache # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.schema import AIMessage, HumanMessage, SystemMessage, chat # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


class Scientist(BaseModel):
    name: str = Field(description='Name of a Scientist')
    discoveries: list = Field(description="Python list of discoveries")

parser = PydanticOutputParser(pydantic_object=Scientist)
print(parser.get_format_instructions())

human_prompt = HumanMessagePromptTemplate.from_template("{request}\n{format_instructions}")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

request = chat_prompt.format_prompt(request="Tell me about a famous scientist",
                                    format_instructions=parser.get_format_instructions()).to_messages()

result = llm.invoke(request,temperature=0)