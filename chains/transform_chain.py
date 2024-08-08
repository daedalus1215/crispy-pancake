from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from ollama_factory import OllamaFactory

llm = OllamaFactory().create_llm()

yelp_review = open('data/yelp_review.txt').read()
print(yelp_review.split('REVIEW:')[-1].lower())

def transformer_fun(inputs:dict) -> dict: 
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output': lower_case_text}

transform_chain = TransformChain(input_variables=['text'],
                                 output_variables=['output'])

template = 'Create a one sentence summary of this review:\{review}'

OllamaFactory.create_chat_prompt(template)
summary_chain = LLMChain(llm=llm, prompt=prompt, output_key='review_summary')

sequential_chain = SimpleSequentialChain(chains=[transform_chain, summary_chain],
                                         verbose=True)

result = sequential_chain(yelp_review)