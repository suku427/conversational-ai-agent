import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import respond_to_query

app = FastAPI(title="AutoStream Conversational AI", version="1.0.0")


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    response_text = respond_to_query(request.prompt)
    return ChatResponse(response=response_text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
