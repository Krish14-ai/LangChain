import streamlit as st 
from langchain_google_genai import ChatGoogleGenerativeAI 
import streamlit as st 
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")
st.header("Research Tool")

paper_input = st.selectbox("Enter the namme of the research paper ",["Atttention is all you need", "BERT", "GPT-3 language models are Few-short Learners"])

style_input = st.selectbox("enter the type of Summary you like",["Beginner-Friendly", "Technical", "Code-Oriented"])

length_input = st.selectbox("No of Lines", ["3-4 lines short notes", "5-10 line Explanation ", "10-15 line Detailed Explanation"])

        

## creating a Prompt Template 
template = load_prompt("template_1.json")

## Fill the place holders in "input_varianbles"
prompt = template.invoke({
        'paper_input' : paper_input,
        'style_input' :style_input,
        'length_input' : length_input
        })

if st.button("submit") : 
        result = model.invoke(prompt)
        st.write(result.text)