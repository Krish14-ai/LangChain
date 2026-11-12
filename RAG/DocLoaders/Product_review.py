from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model="llama3.2:3b",
    temperature = 1.4
)

prompt = PromptTemplate(
    template= 'answer the following Question\n {question} from the following text \n :- {text}',
    input_variables= ['questions', 'text']
)

parser = StrOutputParser()

url = r'https://www.amazon.in/CUCKOO-Respure-Copper-Pollution-Allergies/dp/B0DK1T72LR/ref=sr_1_1_sspa?adgrpid=1324913117401771&dib=eyJ2IjoiMSJ9.f2WHXu6qc79jIuRsPE9SSJvI4xd1ub8UmWxeE-_eHPE_jMemULDMzMCW8LxUVwhiGuekleqI87hYNPjvZzf0g3YmL5OOaj1cHbkp-qKGHOPVdC_Y2QmMN_MMceB4WYjPJLUg2Lvuq0KUP1DNbY24G8NyVHFcEAUZmidPfycmseQkQ9g4-xGO1PDZNbKCx2DC.7EKRywMQxMGLd_vNzUaIeWHeIFlvnwCEFtioLARB3f0&dib_tag=se&hvadid=82807340862342&hvbmt=be&hvdev=c&hvlocphy=155043&hvnetw=o&hvqmt=e&hvtargid=kwd-82807950945301%3Aloc-90&hydadcr=24998_2367461&keywords=amazon+company&mcid=8722d031a0a7302d9bd1779e7eaf0a8c&msclkid=00fb31bab7d91e8b8fca5b4c91c3687e&qid=1794516509&sr=8-1-spons&aref=quKuptLpL9&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question' : 'what is the product that has been listed in the site', 'text' : loader.load()})

print(result)