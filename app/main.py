"""FastAPI service for the SHL Assessment Recommender.

Two endpoints:
  GET /health  -> {"status": "ok"}
  POST /chat   -> stateless conversational reply + grounded shortlist

The agent and retriever are loaded once at startup. We keep the
process stateless so the same container can serve any conversation 
without per-session memory.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .agent import Agent, ChatResponse
from .catalog import Catalog
from .llm import configure_api_key
from .retriever import Retriever


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("content")
    @classmethod
    def _content_max_len(cls, v: str) -> str:
        # Defensive: very long pasted text is fine, but cap at 16KB to keep prompt size sane.
        if len(v) > 16_000:
            return v[:16_000]
        return v


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponseModel(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

_state: dict = {}


@app.on_event("startup")
def _load_state() -> None:
    # Multi-key support is handled inside app.llm at import time, but we still
    # want to honour a single GEMINI_API_KEY if that's the only one set.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        configure_api_key(key)
    catalog = Catalog.load()
    retriever = Retriever.load(catalog)
    _state["agent"] = Agent(catalog, retriever)
    _state["catalog_size"] = len(catalog)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponseModel)
def chat(req: ChatRequest) -> ChatResponseModel:
    agent: Agent | None = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="agent not initialized")
    raw_messages = [m.model_dump() for m in req.messages]
    result: ChatResponse = agent.respond(raw_messages)
    # Final schema enforcement.
    recs = [Recommendation(**r) for r in result.recommendations[:10]]
    return ChatResponseModel(
        reply=result.reply,
        recommendations=recs,
        end_of_conversation=bool(result.end_of_conversation),
    )
