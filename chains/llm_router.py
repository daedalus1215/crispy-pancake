from ollama_factory import OllamaFactory
from langchain.chains import LLMChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain


# Student ask Physics. We will have an llm decide if the question is advance or not.
# Question #1: 'How does a magnet work?'
# Question #2: 'Explain Feynman diagram?'
# Flow: INPUT --> ROUTER --> LLM DECIDES CHAIN --> CHAIN --> OUTPUT

llm = OllamaFactory().create_llm()

# Begin to create the 2 options our LLM Router will choose from.
prompt_infos = [
    {'name': 'beginner physics',
     'description': 'Answers basic physics questions',
     'template': 'You are a physics teacher who is really focused on beginners and explaining complex concepts in simple to understand terms. You assume no prior knowledge. Here is your question:\n{input}',
     },
    {'name': 'advanced physics',
     'description': 'Answers advanced physics questions',
     'template':'You are a physics professor who explains physics topics to advanced audience members. You can assume anyone you answer has a PhD in Physics. Here is your question:\n{input}'
     }
]

# Load up our chain models in the required format for Langhcain.
destination_chains = {}
for p in prompt_infos:
    destination_chains[p['name']] = LLMChain(llm=llm, 
                                             prompt=OllamaFactory.create_chat_prompt(p['template']))

# Setup the LLMChain template
default_chain = LLMChain(llm=llm, 
                         prompt=OllamaFactory.create_chat_prompt('{input}'))

# Can look at the format required for Langchain's LLM Router.
# print(MULTI_PROMPT_ROUTER_TEMPLATE)

# Setup the strings in the routing template format
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
# print(destinations)q
destination_str = "\n".join(destinations)
# print(destination_str)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destination_str)
# print(router_template)
router_prompt = PromptTemplate(template=router_template,
                               input_variables=['input'],
                               output_parser=RouterOutputParser())

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True)

chain.run("What is Feynmans Diagram")
