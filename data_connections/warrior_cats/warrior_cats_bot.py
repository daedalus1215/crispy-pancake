import pandas as pd
from langchain_community.vectorstores import Chroma

def query_warrior_cats_db(question):
    db_path = './warrior_cats_db.db'
    doc = Chroma.load(db_path, embedding_function_name="all-MiniLM-L6-v2")
    
    # Use the ContextualCompressionRetriever to find relevant documents
    llm = Ollama(
        model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, 
                                                       base_retriever=doc.as_retriever())
    
    compressed_docs = compression_retriever.get_relevant_documents(question)
    
    # Print the most relevant document content
    print(compressed_docs[0].page_content)

# Example usage
query_warrior_cats_db("Who shouted 'Sandpaw!'?")