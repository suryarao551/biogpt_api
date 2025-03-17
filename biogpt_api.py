from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load BioGPT model
pipe = pipeline("text-generation", model="microsoft/biogpt")

# Initialize FastAPI app
app = FastAPI()

# Define request format
class TextInput(BaseModel):
    text: str
    max_length: int = 100  # Optional parameter to limit output length

@app.post("/generate")
async def generate_text(input: TextInput):
    """Generates text using BioGPT given an input prompt."""
    generated = pipe(input.text, max_length=input.max_length)[0]["generated_text"]
    return {"input": input.text, "output": generated}

# Root endpoint
@app.get("/")
def home():
    return {"message": "BioGPT API is running. Use /generate to generate text."}
