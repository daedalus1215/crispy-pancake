from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore

class OllamaAdapter:
    def __init__(self):
        llm = Ollama(
            model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    
    