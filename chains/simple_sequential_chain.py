from ollama_factory import OllamaFactory
from langchain.chains import LLMChain, SimpleSequentialChain

llm = OllamaFactory().create_llm()

# --> topic blog post 
# --> outline 
# --> create blog post from outline 
# -->  blog post text

chain_one = LLMChain(llm=llm, 
                     prompt=OllamaFactory.create_chat_prompt("Give me a simple bullet point outline for a blog post on {topic}"))

chain_two = LLMChain(llm=llm, 
                     prompt=OllamaFactory.create_chat_prompt("Write a blog post using this outline {outline}"))

full_chain = SimpleSequentialChain(chains=[
                                           chain_one, 
                                           chain_two], 
                                   verbose=True)

result = full_chain.run('Cheesecake')