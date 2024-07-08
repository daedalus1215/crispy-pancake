from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.cache import InMemoryCache # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.schema import AIMessage, HumanMessage, SystemMessage, chat # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser



llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

parser = CommaSeparatedListOutputParser()

human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
system_prompt = SystemMessagePromptTemplate.from_template("You always reply to questions only in datetime patterns. ")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

model_request = chat_prompt.format_prompt(request='write a poem about animals',
                          format_instructions=parser.get_format_instructions()).to_messages()

result = llm.invoke(model_request)




output_parser = DatetimeOutputParser()

output_parser.get_format_instructions()


template_text = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
result = chat_prompt.format_prompt(request="what date was the 13th amendment ratified in the US?",
                          format_instructions=output_parser.get_format_instructions()).to_messages()


from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
new_parser.parse(result)

