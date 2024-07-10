import datetime
from langchain.callbacks.manager import CallbackManager # type: ignore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # type: ignore
from langchain_community.llms import Ollama # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore



class HistoryQuiz:  
    
    def __init__(self):
        self.quizzes = []
        self.llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        
    def create_history_question(self, topic):
        return self.llm.invoke(PromptTemplate(input_variables=["topic"], 
                              template='Can you provide a historical question for {topic}'))
    
    def get_AI_answer(self, question):
        '''
        This method should get the a nswer to the historical question from the method above.
        Note: This answer must be in datetime format! Use DateTimeOutputParser to confirm!
        
        September 2, 1945 --> datetime.datetime(1945, 2, 0, 0)
        '''
        return self.llm.invoke(PromptTemplate(input_variable=["question"], template='{question}'))
    
    def get_user_answer(self):
        '''
        //@TODO: we will get the user's input in a traditional way
        '''
        while True:
            try:
                date_str = input("Enter the date (YYYY-MM-DD): ")
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return dt.date()
            except ValueError:
                print("Invalid input. Please try again.")
    
    # def check_user_answer(self, user_name, ai_answer):
    #     '''
    #     Should check the user answer against the AI answer and return the difference between
    #     '''
    #     pass