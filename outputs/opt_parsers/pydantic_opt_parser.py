from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)