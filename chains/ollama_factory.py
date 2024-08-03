from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore

class OllamaFactory:

    def create_llm(self):
        return Ollama(
            model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    
    def create_human_prompt(message):
        return HumanMessagePromptTemplate.from_template(message)
    
    def convert_human_prompt_to_chat_prompt(human_message_prompt_template):
        return ChatPromptTemplate.from_messages([human_message_prompt_template])
    
    def create_chat_prompt(message):
        return ChatPromptTemplate.from_template(message)
    