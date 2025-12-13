import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ka.agent import AgentLoop, Tools
from ka.rag import Retriever


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[Dict[str, Any]]


def create_app(
    index_dir: str = "dataset/index",
    notes_path: str = "dataset/processed/notes.jsonl",
) -> FastAPI:
    app = FastAPI(title="Knowledge Assistant", version="0.2")

    index_dir = os.path.abspath(os.path.expanduser(index_dir))
    notes_path = os.path.abspath(os.path.expanduser(notes_path))

    retriever = Retriever(index_dir=index_dir)
    tools = Tools(retriever=retriever, notes_path=notes_path)
    agent = AgentLoop(tools=tools)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        answer, calls = agent.run(req.question)
        return AskResponse(
            answer=answer,
            tool_calls=[{"name": c.name, "args": c.args} for c in calls],
        )

    return app


app = create_app()


