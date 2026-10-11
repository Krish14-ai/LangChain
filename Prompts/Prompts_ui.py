import streamlit as st 
from langchain_google_genai import ChatGoogleGenerativeAI 
import streamlit as st 
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")
st.header("Research Tool")

paper_input = st.selectbox("Enter the namme of the research paper ",["Atttention is all you need", "BERT", "GPT-3 language models are Few-short Learners"])

style_input = st.selectbox("enter the type of Summary you like",["Beginner-Friendly", "Technical", "Code-Oriented"])

length_input = st.selectbox("No of Lines", ["3-4 lines short notes", "5-10 line Explanation ", "10-15 line Detailed Explanation"])

## creating a Prompt Template 
template = PromptTemplate(
        
            template = f""" Please summarize the research paper titled "{paper_input}" with the following specifications:
    
                    Explanation Style: {style_input}
                    Explanation Length: {length_input}
    
                    1. **Mathematical Details:**
                    - Include relevant mathematical equations if present in the paper.
                    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    
                    2. **Analogies:**
                    - Use relatable analogies to simplify complex ideas.
    
                    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
    
                    Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
            input_variables = ['paper_input','style_input','length_input']
    
)

## Fill the place holders in "input_varianbles"
prompt = template.invoke({
    'paper_input' : paper_input,
    'style_input' :style_input,
    'length_input' : length_input
})

if st.button("submit") : 
        result = model.invoke(prompt)
        st.write(result.text)

