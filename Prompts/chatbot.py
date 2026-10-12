from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash"
)

st.header("ChatBot")

while True:
    quiery = st.text_input("Enter any Quiery : ")
    if st.button("enter") : 
    
        if quiery == "Exit":
            break
        
        result = model.invoke(quiery)
        
        st.write(result.text)
    