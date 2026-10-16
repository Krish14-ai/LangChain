from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-3.7-flash")

chat_history = [
    SystemMessage(content = "You are a Helpful AI Assistant")
]

while True:
    user_input = input('you :- ')
    