from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    'google-genai:gemini-3.5-flash'
)

parser = JsonOutputParser()

template = PromptTemplate(
    template= 'give me the name,age, city of a fictional person\n{format_instruction}',
    input_variables=['format_instruction'],
    ## instructions to the LLM to return the output in JSON format, also its value is filled before runtime, that's why we call it a Partial Variable
    partial_variables= {'format_instruction' : parser. get_format_instructions()}
)

prompt = template.format()

print(prompt)