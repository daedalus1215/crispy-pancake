import pandas as pd
from langchain_community.vectorstores import Chroma
import chromadb
import fitz  # PyMuPDF for PDF handling
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore
import pdfplumber

def warrior_cat_helper(question):
    db_path = './warrior_cats_db.db'
    
    # PART ONE:
    # Load the Warrior Cats books as PDFs
    pdf_files = [
        "../../warrior_cats/01_into_the_wild.pdf",
        "../../warrior_cats/02_fire_and_ice.pdf",
        "../../warrior_cats/03_forest_of_secrets.pdf",
        # Add more files as needed
    ]

    compressed_docs = []
    
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()
        
        # embedding_function = Chroma.default_embedding_function("all-MiniLM-L6-v2")
        client = chromadb.HttpClient(host='localhost', port=8000)
        doc = Chroma.load([text], client)
        doc.persist(db_path)

        llm = Ollama(
            model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        compressor = LLMChainExtractor.from_llm(llm)

        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                       base_retriever=doc.as_retriever())
        
        compressed_docs.append(compression_retriever.get_relevant_documents(question)[0])

    return compressed_docs

def query_warrior_cats_db(question):
    compressed_docs = warrior_cat_helper(question)
    
    # Print the most relevant document content
    for i, doc in enumerate(compressed_docs):
        print(f"Relevant Document {i+1}:")
        print(doc.page_content)

# Example usage
query_warrior_cats_db("Who shouted 'Sandpaw!'?")