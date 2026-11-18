from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class Person(BaseModel):
    
    name : str = Field(description="Enter the name of the person : ")
    age : int = Field(description="Enter the age of the person : ")
    nationality : str = Field(description= "Nationality of this person: ")
    city : str = Field(description= "Where does the person came form : ")

parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
    template  = "Generate the name,age,nationality,city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'place' : 'Indian'})

print(prompt,'\n\n')
## =============================================
## normal way
## =============================================


# result = model.invoke(prompt)

# final_result = parser.parse(result.text)

# print(final_opt)

## =============================================
## Chaining Method
## =============================================

chain = template | model  | parser 

final_result = chain.invoke([])

print(final_result)
