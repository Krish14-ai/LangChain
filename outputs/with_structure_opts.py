from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)

# Schema
class Review(BaseModel): 
    key_themes  : list[str] = Field( description="Write down all the key themes discussed in the review in a list")
    
    summary : str= Field( description="A brief summary of the review")
    
    sentiment : Literal['positive', 'Negative'] =Field( description="the sentiment of the review")
    
    pros : Optional[list[str]] = Field( description="Extract all the Pros of the Comments in a list", default=None)
    
    cons : Optional[list[str]]= Field(description="Extract all the Cons of the Comments in a list", default= None)
    
    name : Optional[str] = Field(default= None, description= "name of the Reviewer")
structured_model = model.with_structured_output(Review)

result= structured_model.invoke("This Nike works considerably well. It secretly improves my basketball by a lot.")

print(result)

print(type(result))