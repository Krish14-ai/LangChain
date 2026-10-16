from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model = "gemini-3.7-flash")

