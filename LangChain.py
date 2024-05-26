from langchain_community.llms import Ollama

# llm = Ollama(model="llama2")
# llm.invoke("The first man on the moon was ...")



from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
llm.invoke("The first man on the moon was ...")