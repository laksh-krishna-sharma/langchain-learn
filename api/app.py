from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langserve import add_routes
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import uvicorn

# Load environment variables
load_dotenv()

# Set environment variables for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# FastAPI app
app = FastAPI(
    title="Langchain With Gemma 3 API",
    version="1.0.0",
    description="A simple API server using Langchain's Runnable interfaces",
)

# Define input model (Pydantic v2-compatible)
class BotInput(BaseModel):
    question: str

# Initialize model and prompt
llm = OllamaLLM(model="gemma3:1b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}"),
])

chain = prompt | llm

add_routes(
    app,
    chain.with_types(input_type=BotInput, output_type=str),
    path="/bot"
)

# Run app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
