from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_community.llms import Ollama
from langserve import add_routes
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain With Gemma 3 API",
    version="1.0.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

llm = Ollama(model="gemma3:1b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}"),
    ]
)

chain = prompt | llm

add_routes(
    app,
    chain.with_types(input_type=dict, output_type=str),
    path="/bot"
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)