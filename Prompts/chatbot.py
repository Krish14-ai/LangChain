from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

st.header("ChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# replay history on every rerun
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# each submitted message triggers a rerun, which acts as the loop
if query := st.chat_input("Enter any Query"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # send the full history so the model remembers context
    result = model.invoke(st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": result.text})
    with st.chat_message("assistant"):
        st.write(result.text)