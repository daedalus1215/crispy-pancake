from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.schema import AIMessage,HumanMessage,SystemMessage # type: ignore

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

human_message = HumanMessage(content='Tell me a fact about Pluto')
system_message = SystemMessage(content='you are a very rude teenager who won\t answer')
#llm.invoke("Here is a fun fact about Pluto:")

# Using __call__ directly
results = llm.invoke([system_message, human_message]) 

print(results)
