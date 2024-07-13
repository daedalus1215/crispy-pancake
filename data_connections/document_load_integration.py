from langchain_community.document_loaders import HNLoader
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore

loader = HNLoader("https://news.ycombinator.com/item?id=40952215")
data = loader.load()
print(data[0].page_content)
print(data[0].metadata)

human_prompt = HumanMessagePromptTemplate.from_template("Please give me a short summary of the following HackerNews comment: \n{comment}")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])


llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

llm.invoke(chat_prompt.format_prompt(comment=data[0].page_content).to_messages())