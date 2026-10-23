from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

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

# Generate the report
prompt1 = template.invoke({"topic": "black holes"})
result = model.invoke(prompt1)

# Convert the model output into a plain string
report = parser.invoke(result)

# Pass the report to the second prompt
prompt2 = template2.invoke({"text": report})

# Generate the summary
summary = model.invoke(prompt2)

# Parse the final output into a string
final_summary = parser.invoke(summary)

print(final_summary)
