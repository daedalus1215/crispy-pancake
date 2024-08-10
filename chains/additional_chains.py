from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from ollama_factory import OllamaFactory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# docs
llm = OllamaFactory().create_llm()
embedding_function = OllamaFactory.create_embedding_function()
db_connection = OllamaFactory.create_db_connection('../data_connections/data/US_Constitution',
                                   embedded_function=embedding_function)

chain = load_qa_chain(llm, chain_type='stuff') # stuff = inserting context into our model (pulled later on from choma store)
question = "What is the 15th amendment?"
docs = db_connection.similarity_search(question)
chain.run(input_documents=docs, question=question)