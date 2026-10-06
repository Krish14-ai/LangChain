from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model  = "sentence-transformers/all-MiniLM-L6-v2")

## Embedding docs 
docs = [
    "Delhi is the Capital of India",
    "Hyderabad is the Capital of Telangana",
    "Lucknow is the Capital of UttarPradesh"
]

vectors = embeddings.embed_documents(docs)

print(str(vectors))