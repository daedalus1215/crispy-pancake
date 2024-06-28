from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.cache import InMemoryCache # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage, chat

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

system_template = "You are an AI recipe assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template="{recipe_request}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# grab the inputs 
print(system_message_prompt.input_variables)
# grab the inputs
print(human_message_prompt.input_variables)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(chat_prompt.input_variables)
llm.invoke(chat_prompt.format_prompt(cooking_time='60 min', 
                          recipe_request="Quick Snack",
                          dietary_preference="Vegan").to_messages())