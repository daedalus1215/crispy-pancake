from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.chains import LLMChain

from ollama_factory import OllamaFactory

human_prompt = HumanMessagePromptTemplate.from_template('Make up a funny company name, for a company that makes: {product}')
chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt]);

chat = OllamaFactory().create_llm()

chain = LLMChain(llm=chat, prompt=chat_prompt_template)

chain.run(product='Computers')


