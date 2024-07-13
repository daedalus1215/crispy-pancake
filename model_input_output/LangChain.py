from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore

#llm.invoke("Here is a fun fact about Pluto:")
prompts = ['Please finish this statement: "Here is a fun fact about Pluto:"']

# Using __call__ directly
results = llm.invoke(prompts) 

print(results)  
