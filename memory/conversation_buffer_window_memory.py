from ollama_factory import OllamaFactory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory 


# HUMAN and AI
# k = 1 WINDOW --> k interactions

llm = OllamaFactory().create_llm()
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input='Hello how are you?')
conversation.predict(input='tell me a math fact')
conversation.predict(input='tell me a fact about Mars')

# print out the list of messages here:
print(memory.buffer)

memory.load_memory_variables({})

