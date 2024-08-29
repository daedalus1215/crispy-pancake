from ollama_factory import OllamaFactory
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory 

# HUMAN and AI
# k = 1 WINDOW --> k interactions

llm = OllamaFactory().create_llm()
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
# Human AI Human AI --> LLM Summary --> 
conversation = ConversationChain(llm=llm, memory=memory)
conversation.predict(input='Give me some travel plans for San Francisco')
conversation.predict(input='Give me some travel plans for New York City')

memory.load_memory_variables({})