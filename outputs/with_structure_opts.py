from langchain.chat_models import init_chat_model 
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.8-flash'
)