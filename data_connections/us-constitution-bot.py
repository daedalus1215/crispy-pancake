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
# Compressions
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

def us_constitution_helper(question):
    '''
    Takes in a question about the US constitution and returns the most relevant
    part of the constitution. Notice it may not directly answer the actual question!

    Follow the steps below to fill out this function:
    '''
    # PART ONE:
    # LOAD the us constitution file
    loader = TextLoader("data/US_Constitution.txt")
    documents = loader.load()

    # PART TWO
    # Split the document into chunks (you choose how and what size)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)
    
    # PART THREE
    # EMBED THE Documents (now in chunks) to a persisted ChromaDB
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function, persist_directory='./solution')
    db.persist()

    # PART FOUR
    # Use LLM and ContextualCompressionRetriever to return the most 
    # Relevant part of the documents.
    llm = Ollama(
        model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                       base_retriever=db.as_retriever())
    compressed_docs = compression_retriever.get_relevant_documents(question)
    print(compressed_docs[0].page_content)


us_constitution_helper("What is the 1th Amendment?")