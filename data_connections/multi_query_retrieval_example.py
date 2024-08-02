from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama # type: ignore
import logging 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore

loader= WikipediaLoader(query='MKUltra')
documents = loader.load()
print(documents)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=519)
docs = text_splitter.split_documents(documents)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs,embedding_function,persist_directory="./some_new_mkultra")
db.persist()

question = "When was this declassified?"

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

retriever_from_llm.get_relevant_documents(query=question)