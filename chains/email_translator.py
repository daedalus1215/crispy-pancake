from ollama_factory import OllamaFactory
from langchain.chains import LLMChain, SequentialChain

spanish_email = open('./data/spanish_customer_email.txt').read()
print("\noriginalEmail\n")
print(spanish_email)
print("\n")


def translate_and_summarize(email):
    llm = OllamaFactory().create_llm()
    # detect the language
    prompt = OllamaFactory.create_chat_prompt("Return the language this email is written in:\{email}.\n Only return the language it is in.")
    chain1 = LLMChain(llm=llm, prompt=prompt,output_key="language")
    
    # translate
    translate_prompt = OllamaFactory.create_chat_prompt("Translate this email from {language} to English.\n"+email)
    chain2 = LLMChain(llm=llm, prompt=translate_prompt, output_key="translated_email")
    
    # summary
    template_summary = OllamaFactory.create_chat_prompt("Create a short summary of this email:\n{translated_email}")
    chain3 = LLMChain(llm = llm, prompt = template_summary, output_key="summary")
    
    seq_chain = SequentialChain(chains=[chain1, chain2, chain3],
                                input_variables=['email'],
                                output_variables=['language', 'translated_email', 'summary'], 
                                verbose=True)
    
    return seq_chain(email)

result = translate_and_summarize(spanish_email)
print('\nresult keys\n')
print(result.keys())
print('\nlanguage\n')
print(result['language'])
print('\ntranslated_email\n')    
print(result['translated_email'])
print('\nsummary\n')    
print(result['summary'])
