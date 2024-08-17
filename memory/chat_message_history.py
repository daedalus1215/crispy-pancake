from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("Hello nice to meet you")
history.add_ai_message("Nice to meet you!")

print(history.messages)