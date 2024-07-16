from langchain.text_splitter import CharacterTextSplitter

with open('data/FDR_State_of_Union_1944.txt') as file:
    speech_text = file.read()

print(len (speech_text))

print(len(speech_text.split())) 

text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
texts = text_splitter.create_documents([speech_text])
print(type(texts[0]))
print(texts[0])
