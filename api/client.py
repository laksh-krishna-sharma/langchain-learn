import requests
import streamlit as st

def get_ollama_response(input_text):
    # Properly send JSON payload
    response = requests.post(
        'http://localhost:8000/bot/invoke',
        json={
            "input": {
                "question": input_text
            }
        }
    )

    if response.status_code == 200:
        return response.json()['output']
    else:
        return "Error: " + response.text

st.title('Langchain With Gemma 3 API')
input_text = st.text_input("Search the topic you want")

if input_text:
    st.write(get_ollama_response(input_text))
