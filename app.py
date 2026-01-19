
import streamlit as st

st.title("My First Streamlit App")
st.write("Hello, this is running from a Jupyter Notebook!")

user_input = st.text_input("Enter your name:", "Streamlit User")
st.write(f"Welcome, {user_input}!")
