from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)

# Schema
class Review(TypedDict): 
    key_themes  : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "the sentiment of the review"]
    pros : Annotated[list[str], "Extract all the Pros of the Comments in a list"]
    cons : Annotated[list[str],"Extract all the Cons of the Comments in a list"]
    
structured_model = model.with_structured_output(Review)

result= structured_model.invoke("This Nike works considerably well. It secretly improves my basketball by a lot.")

print(result)

print(type(result))