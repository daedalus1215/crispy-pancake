from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ollama_factory import OllamaFactory
import pickle

llm = OllamaFactory().create_llm()
memory = ConversationBufferMemory()
conversation = ConversationChain(llm = llm, memory = memory, verbose=True)
conversation.predict(input='Hello, nice to meet you')

conversation.predict(input="Tell me about an interesting physic facts")

memory.load_memory_variables({})

# Display the conversation memory
conversation.memory

pickled_str = pickle.dumps(conversation.memory)

with open('convo_memory.pkl', 'wb') as f:
    f.write(pickled_str)

llm = OllamaFactory().create_llm()

new_memory_loaded = pickle.loads(open('convo_memory.pkl', 'rb').read())
reload_conversation = ConversationChain(llm=llm, memory=new_memory_loaded)
print(reload_conversation.memory_buffer)