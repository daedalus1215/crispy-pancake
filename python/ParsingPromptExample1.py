from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

parser = CommaSeparatedListOutputParser()

human_template = "{request}\n{format_instructions}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

model_request = chat_prompt.format_prompt(request='write a poem about animals',
                          format_instructions=parser.get_format_instructions()).to_messages()

result = llm.invoke(model_request)

