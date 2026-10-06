from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", output_dimensionality= 300)

docs = [
    "Delhi is the Capital of India",
    "Hyderabad is the Capital of Telangana",
    "Lucknow is the Capital of UttarPradesh"
]

query = "What is the capital of India"

docs_embeddings = embedding.embed_documents(docs)
query_embed = embedding.embed_query(query)

socres = cosine_similarity([query_embed], docs_embeddings)[0]

## 1) Enumerate will give indexes to the scores,
## 2) List will give one single list
## 3) Sorted will sort the scores
## 4) lambda function will use the 2nd index as a key to sort and [-1] will give me the biggest number 
index,score  = sorted(list(enumerate(socres)),key = lambda x : x[1])[-1]

print(query)
print(docs[index])
print("Similarity Score is : ",score)