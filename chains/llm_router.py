from ollama_factory import OllamaFactory
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# Student ask Physics. We will have an llm decide if the question is advance or not.
# Question #1: 'How does a magnet work?'
# Question #2: 'Explain Feynman diagram?'
# Flow: INPUT --> ROUTER --> LLM DECIDES CHAIN --> CHAIN --> OUTPUT

llm = OllamaFactory().create_llm()

beginner_template = 'You are a physics teacher who is really focused on beginners and explaining complex concepts in simple to understand terms. You assume no prior knowledge. Here is your question:\n{input}'
expert_template = 'You are a physics professor who explains physics topics to advanced audience members. You can assume anyone you answer has a PhD in Physics. Here is your question:\n{input}'

prompt_infos = [
    {'name': 'beginner physics',
     'description': 'Answers basic physics questions',
     'template': beginner_template,
     },
    {'name': 'advanced physics',
     'description': 'Answers advanced physics questions',
     'template':expert_template
     }
]

destination_chains = {}
for p in prompt_infos:
    destination_chains[p['name']] = LLMChain(llm=llm, 
                                             prompt=OllamaFactory.create_chat_prompt(p['template']))
