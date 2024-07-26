from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

loader = TextLoader('data/FDR_State_of_Union_1944.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=700)
docs = text_splitter.split_documents(documents)

# docs
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embedding_function, persist_directory="./speech_new_db")

db.persist() # tell chroma to save it.


# pull it out of the database
db_new_connection = Chroma(persist_directory="./speech_new_db", embedding_function=embedding_function)
similar_docs = db_new_connection.similarity_search('What did FDR say about the cost of food law?')
# print(similar_docs[0].page_content)

loader = TextLoader("data/lincoln_state_of_union.txt")
documents = loader.load()
docs = text_splitter.split_documents(documents)
db_new_connection = Chroma.from_documents(docs, embedding_function, persist_directory='./speech_new_db')
blocks = db_new_connection.similarity_search('said of the proportion')

# print(blocks[0].page_content)
# print(blocks[0].metadata)

print(db_new_connection)

retriever = db_new_connection.as_retriever()
results = retriever.get_relevant_documents('cost of food law')
print(results)