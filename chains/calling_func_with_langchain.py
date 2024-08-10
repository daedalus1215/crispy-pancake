from ollama_factory import OllamaFactory
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_core.output_parsers import OpenAIChatOutputParser

class Scientist:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

json_schema = {
    'title': 'Scientist', 
    'description': 'Information about a famous scientist',
    'type': 'object',
    'properties': {
        'first_name': {
            'title': 'First Name',
            'description': 'First name of scientist',
            'type': 'string'
        },
        'last_name': {
            'title': 'Last Name',
            'description': 'Last name of scientist',
            'type': 'string'
        }
    },
    'required': ['first_name', 'last_name']
}

llm = OllamaFactory().create_llm()

# Assuming that create_chat_prompt generates the appropriate chat prompt
chain = create_structured_output_chain(
    json_schema,
    llm,
    OllamaFactory.create_chat_prompt('Name a famous {country} scientist'),
    verbose=True,
    output_parser=OpenAIChatOutputParser()
)

try:
    result = chain.run(country='American')
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
