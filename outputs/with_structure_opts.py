from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)

# Schema
class Review(TypedDict): 
    summary : str 
    sentiment : str
    
structured_model = model.with_structured_output(Review)

result= structured_model.invoke("This Nike works considerably well. It secretly improves my basketball by a lot.")

print(result)

print(type(result))