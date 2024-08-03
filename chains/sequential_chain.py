from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.chains import LLMChain, SequentialChain
from ollama_factory import OllamaFactory

llm = OllamaFactory().create_llm()

# Employee performance review - INPUT TEXT
# review_text --> LLMChain --> Summary
# Summary --> LLMChain --> Weaknesses



chain1 = LLMChain(llm = llm, 
                  prompt=OllamaFactory().create_chat_prompt("Give a summary of this employee's performance review \n {review}"),
                  output_key='review_summary')

chain2 = LLMChain(llm = llm, 
                  prompt=OllamaFactory().create_chat_prompt("Identify key employee weaknesses in this review summary:\n{review_summary}"),
                  output_key='weaknesses')

chain3 = LLMChain(llm = llm, 
                  prompt=OllamaFactory().create_chat_prompt("Create a personalized plan to help address and fix these weaknesses\n{weaknesses}"),
                  output_key='final_plan')

seq_chain = SequentialChain(chains = [chain1, chain2, chain3],
                            input_variables=['review'],
                            output_variables=['review_summary', 'weaknesses', 'final_plan'],
                            verbose=True)

results = seq_chain("""
        Employee Name: Richard
        
        Job Title: Software Engineer
        
        Review Period: [Insert dates, e.g. Q1 2023]
        
        Overall Performance: 4.5/5
        
        Strengths:
        
        
        Consistently delivered high-quality code with attention to detail and adherence to best practices
        
        Demonstrated excellent problem-solving skills, often finding innovative solutions to complex technical challenges
        
        Collaborated effectively with cross-functional teams to drive project success
        
        Showed a strong willingness to learn and adapt to new technologies and processes
        
        
        Areas for Growth:
        
        
        Occasionally struggled with meeting tight deadlines, requiring additional support and prioritization techniques
        
        Could benefit from more formalized testing and debugging procedures to improve code reliability
        
        Opportunities exist to take on more leadership roles or mentorship responsibilities within the team
        
        
        Key Accomplishments:
        
        
        Successfully led development of [project/initiative name], resulting in [desirable outcome, e.g. increased user engagement]
        
        Contributed to significant improvements in [specific metric or KPI] through optimized code and process enhancements
        
        Participated in code reviews and provided constructive feedback to peers
""")