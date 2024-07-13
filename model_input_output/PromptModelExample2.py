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

def prompting_prompter(interest, budget):
    system_template = "You are an AI travel assistant that specializes in {interest} hobbys, can you recommend something less than {budget}?"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    # grab the inputs 
    print(system_message_prompt.input_variables)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    print(chat_prompt.input_variables)
    llm.invoke(chat_prompt.format_prompt(interest=interest, budget=budget).to_messages())

def main():
    prompting_prompter("Hiking", "$10,000")

if __name__ == "__main__":
    main()