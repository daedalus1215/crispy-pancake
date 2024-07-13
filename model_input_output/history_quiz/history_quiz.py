import datetime
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate # type: ignore
from langchain.output_parsers import DatetimeOutputParser



class HistoryQuiz:  
    
    def __init__(self):
        self.quizzes = []
        self.llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        
    def create_history_question(self, topic):
        # return self.llm.invoke(PromptTemplate(input_variables=["topic"], 
        #                       template='Can you provide a historical question for {topic}'))
        system_template = "You write single quiz questions about {topic}. You only return the quiz question."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = "{question_request}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        q = 'Give me a quiz question where the correct answer is a specific date'
        request = chat_prompt.format_prompt(topic=topic, question_request=q).to_messages()
        chat = self.llm.invoke(request)
        return chat
        
    
    def get_AI_answer(self, question):
        # Datetime Parser
        output_parser = DatetimeOutputParser()
        
        # System Template
        system_template = "You answer quiz questions with just a date"
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        # Human template
        human_template = """Answer the user's question: 
        {question}
        {format_instructions}
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        # Compile ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        # Insert question and get_format_instructions
        format_instructions = output_parser.get_format_instructions()
        request = chat_prompt.format_prompt(question=question, format_instructions=format_instructions).to_messages()
        # Chatbot result --> formate date time
        result = self.llm.invoke(request)
        correct_datetime = output_parser.parse(result)
        return correct_datetime
    
    def get_user_answer(self, question):
        '''
       
        '''
        print (question)
        print('\n')
        
        year = int(input("Enter the year: "))
        month = int(input("Enter the month (1-12):"))
        day = int(input("Enter the day (1-31): "))
        
        user_datetime = datetime.datetime(year, month, day)
        return user_datetime
    
    def check_user_answer(self, user_answer, ai_answer):
        print("The difference between answer and your guess: ", str(user_answer - ai_answer))
    

quiz = HistoryQuiz()
question = quiz.create_history_question(topic="Neil Armstrong set foot moon date")
ai_answer = quiz.get_AI_answer(question=question)
user_answer = quiz.get_user_answer(question=question)
quiz.check_user_answer(user_answer=user_answer, ai_answer=ai_answer)