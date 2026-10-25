from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    'google-genai:gemini-3.5-flash'
)