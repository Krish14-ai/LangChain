from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(model="gemini-3.8-flash")

result = llm.invoke("hi")

print(result)
