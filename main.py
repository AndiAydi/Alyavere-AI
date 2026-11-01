
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAIError

from ai import generate_response, conversations

app = FastAPI(title="Alyuvere")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(
        default="guest",
        min_length=3,
        max_length=64,
    )
    mode: Literal["companion", "tutor", "both"] = "both"
    learning_level: Literal[
        "beginner",
        "intermediate",
        "advanced",
    ] = "beginner"


class ChatResponse(BaseModel):
    reply: str
    user_id: str
    mode: str


@app.get("/")
def home():
    return {
        "status": "online",
        "name": "My AI Companion & Tutor",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        reply = generate_response(
            message=request.message,
            user_id=request.user_id,
            mode=request.mode,
            learning_level=request.learning_level,
        )

    except RuntimeError as error:
        if "AI_API_KEY" in str(error):
            raise HTTPException(
                status_code=503,
                detail="AI API key is not configured.",
            ) from error

        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid response.",
        ) from error

    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail="Groq API request failed. Check API access.",
        ) from error

    return ChatResponse(
        reply=reply,
        user_id=request.user_id,
        mode=request.mode,
    )


@app.delete("/chat/{user_id}")
def reset_chat(user_id: str):
    conversations.pop(user_id, None)

    return {
        "status": "success",
        "message": "Conversation memory cleared.",
    }
