from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.document_loaders import CSVLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader



# LOAD DOCUMENT --> SPLIT INTO CHUNK


# EMBEDDING --> EMBED CHUNKS --> VECTORS

# VECT OR CHUNKS --> SAVE CHROMADB

# "query" --> Similarity search chromadb


loader = TextLoader('data/FDR_State_of_Union_1944.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# docs
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


db = Chroma.from_documents(docs, embedding_function, persist_directory="./speech_new_db")

db.persist() # tell chroma to save it.
