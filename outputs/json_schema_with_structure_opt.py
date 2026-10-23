from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import json 

load_dotenv()

model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)

path = r"C:\Users\Krish\Downloads\LangChain\outputs\json_schema.json"

schema = {}
with open(path,'r') as f : 
    schema = json.load(f)

