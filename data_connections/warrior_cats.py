import fitz  # PyMuPDF for PDF handling
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.llms import Ollama # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain.callbacks.manager import CallbackManager # type: ignore

# Helper function to load PDF files and extract text
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

# Helper function to load PDF files and extract text
def load_pdfs_to_text(file_paths):
    documents = []
    for file_path in file_paths:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        documents.append({"page_content": text, "metadata": {"source": file_path}})
    return documents

def warrior_cat_helper(question):
    '''
    Takes in a question about Warrior Cats and returns the most relevant
    part of the book. Notice it may not directly answer the actual question!
    '''
    # PART ONE:
    # Load the Warrior Cats books as PDFs
    pdf_files = [
        "../../warrior_cats/01_into_the_wild.pdf",
        "../../warrior_cats/02_fire_and_ice.pdf",
        "../../warrior_cats/03_forest_of_secrets.pdf",
        # Add more files as needed
    ]
    documents = load_pdfs_to_text(pdf_files)

    # PART TWO
    # Split the document into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)
    
    # PART THREE
    # Embed the documents into ChromaDB
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function, persist_directory='./warrior_cats_db')
    db.persist()

    # PART FOUR
    # Use LLM and ContextualCompressionRetriever to return the most relevant part of the documents
    llm = Ollama(
        model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                       base_retriever=db.as_retriever())
    compressed_docs = compression_retriever.get_relevant_documents(question)

    # Print the most relevant document content
    print(compressed_docs[0].page_content)

# Example usage
warrior_cat_helper("Who shouted 'Sandpaw!'?")
