from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


## What is a Parser ?
## A) An output parser is a tool that converts raw, unstructured outputs from Large language models into structured, usable formats like JSON, lists, CSV, etc....... .


## What is String Opt Parser ?
## A) A String opt Parser is the simplest Parser that is used to Parse the opt of a LLM and return it as a string

## Use Case examples : we ask for a research Paper then we ask it again to summarize it, in this we dont need 
# any other Opt PArser, we can simply use Str Opt Parser

load_dotenv()


model = init_chat_model(
    'google_genai:gemini-3.5-flash'
)

## user Prompt -> LLM -> LLM opt -> Str Parser -> LLM (For a Differnt Task, ex : summarization of a report)

## 1st Prompt -> Detailed report
template = PromptTemplate(
    template="write a  detailed report on {topic}",
    input_variables = ['topic']
    
)

# Initialize the string output parser
parser = StrOutputParser()

## 2nd prompt -> summary
template2 = PromptTemplate(
    template ="Write 5 line summary on the following text.\n\n {text}",
    input_variables= ['text']
)


chain = template | model | parser | template2 | model | parser

result2 = chain.invoke("What is a Black Hole")

final_summary = chain.invoke({"topic": "black holes"})

print(final_summary)