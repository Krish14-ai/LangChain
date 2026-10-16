from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(
    model = "gemini-3.7-flash",
    thinking_level="low",
    max_retries=1
    )

chat_history = [
    SystemMessage(content = "You are a Helpful AI Assistant")
]

while True:
    user_input = input('you :- ')
    chat_history.append(HumanMessage(content = user_input))
    
    if user_input == 'exit':
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result))
    
    print("AI :- ",result)