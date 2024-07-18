
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.document_loaders import CSVLoader

lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
embed_model = LangchainEmbedding(lc_embed_model)

# Basic embedding example
# embeddings = embed_model.get_text_embedding(
#     "It is raining cats and dogs here!"
# )
# print(len(embeddings), embeddings[:10])

loader = CSVLoader("data/penguins.csv")

data = loader.load()    

#print([text.page_content for text in data])

embedded_docs = embed_model.get_agg_embedding_from_queries([text.page_content for text in data])
print(embedded_docs)