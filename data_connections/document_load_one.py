from langchain.document_loaders import CSVLoader, BSHTMLLoader, PyPDFLoader

loader = CSVLoader("data/penguins.csv")
data = loader.load()
type(data[0])

print(data[0].page_content)

html_loader = BSHTMLLoader("data/some_website.html")
data = loader.load()

pdf_loader = PyPDFLoader("data/SomeReport.pdf")

print(pdf_loader.load()[0].page_content.replace('\n', ' '))
