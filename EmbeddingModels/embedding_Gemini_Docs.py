from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2")

documents = [
    "Delhi is the Capital of India",
    "Hyderabad is the Capital of Telangana",
    "Lucknow is the Capital of UttarPradesh"
]

result = embeddings.embed_documents(documents)

print(str(result))