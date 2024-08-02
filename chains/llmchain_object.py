from chains.ollama_adapter import OllamaAdapter
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.chains import LLMChain

human_prompt = HumanMessagePromptTemplate.from_template('Make up a funny company name, for a company that makes: {product}')
chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt]);

chat = OllamaAdapter()

chain = LLMChain(llm=chat, prompt=chat_prompt_template)

chain.run(product='Computers')


