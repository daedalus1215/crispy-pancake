from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ollama_factory import OllamaFactory

llm = OllamaFactory().create_llm()
memory = ConversationBufferMemory()
conversation = ConversationChain(llm = llm, memory = memory, verbose=True)
conversation.predict(input='Hello, nice to meet you')

conversation.predict(input="Tell me about an interesting physic facts")