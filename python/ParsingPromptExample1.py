from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.cache import InMemoryCache # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.schema import AIMessage, HumanMessage, SystemMessage, chat # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


parser = CommaSeparatedListOutputParser()
print(parser.parse("red, blue, green"))


human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

model_request = chat_prompt.format_prompt(request='give me 5 characteristics of dogs',
                          format_instructions=parser.get_format_instructions()).to_messages()

result = llm.invoke(model_request)


