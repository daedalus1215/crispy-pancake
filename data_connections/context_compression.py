from langchain_chroma import Chroma
from langchain_community.llms import Ollama # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore
from  langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = Ollama(
    model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

db_connection = Chroma(persist_directory='./some_new_mkultra', embedding_function=embedding_function)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                       base_retriever=db_connection.as_retriever())
# This gives us too much context
# docs = db_connection.similarity_search("When was this declassified?")
# print(docs)

compressed_docs = compression_retriever.get_relevant_documents("When was this declassified?")
print("HIIIII")
# Still too much context:
# compressed_docs[0].page_content
# Just the right amount of context, because we are using compressed docs and grabbing the summary.
print(compressed_docs[0].metadata['summary'])
