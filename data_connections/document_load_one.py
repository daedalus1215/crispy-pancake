from langchain.document_loaders import CSVLoader, BSHTMLLoader


loader = CSVLoader("data/penguins.csv")
data = loader.load()
type(data[0])

print(data[0].page_content)

loader = BSHTMLLoader("data/some_website.html")
data = loader.load()
