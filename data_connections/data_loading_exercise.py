from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # type: ignore
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_community.document_loaders import WikipediaLoader




def answer_question_about(person_name, question):
    '''
    Use the wikipedia Document Loader to help answer questions about someone,
    insert it as additional helpful context.
    '''
    # load document
    loader = WikipediaLoader(query=person_name, load_max_docs=1) # usually dob is within the first doc
    context_text = loader.load()[0].page_content # grabbing the first item in the page and then grabbing the context
    
    # connect to llm model
    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    
    # prompt - format question
    template = "Answer this question:\n{question}\n Here is some extra context:\n{document}"
    human_prompt = HumanMessagePromptTemplate.from_template(template)
    
    # chat prompt - get results
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    return llm.invoke(chat_prompt.format_prompt(question=question, document=context_text).to_messages())


answer_question_about("Claude Shannon", "When was he born?")

