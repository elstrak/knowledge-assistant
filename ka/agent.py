from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ka.jsonl import read_jsonl
from ka.retriever import Retriever, RetrievalHit
from ka.generator import format_answer


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Dict[str, object]


class Tools:
    def __init__(self, retriever: Retriever, notes_path: str):
        self.retriever = retriever
        self.notes_path = notes_path

    def search(self, query: str, k: int = 5) -> List[RetrievalHit]:
        return self.retriever.retrieve(query=query, k=k)

    def get_note(self, note_id: str) -> Optional[Dict[str, object]]:
        for row in read_jsonl(self.notes_path):
            if str(row.get("id", "")) == note_id:
                return row
        return None


class SimplePlanner:
    """
    Минимальный планировщик (agent):
    - если пользователь просит открыть заметку/показать заметку → get_note
    - иначе → search
    """

    def plan(self, user_input: str) -> ToolCall:
        low = user_input.lower()
        if ("покажи" in low or "открой" in low or "open" in low) and ".md" in low:
            # грубо вытаскиваем note_id как "что-то.md"
            token = _find_md_token(user_input)
            if token:
                return ToolCall(name="get_note", args={"note_id": token})
        return ToolCall(name="search", args={"query": user_input, "k": 5})


class AgentLoop:
    """
    Агентный цикл:
    - план → tool call
    - если плохо, простое query expansion и повторный поиск
    """

    def __init__(self, tools: Tools, planner: Optional[SimplePlanner] = None):
        self.tools = tools
        self.planner = planner or SimplePlanner()

    def run(self, user_input: str, k: int = 5) -> Tuple[str, List[ToolCall]]:
        calls: List[ToolCall] = []
        call = self.planner.plan(user_input)
        if call.name == "search":
            call.args["k"] = int(k)
        calls.append(call)

        if call.name == "get_note":
            note_id = str(call.args["note_id"])
            note = self.tools.get_note(note_id)
            if not note:
                return f"Заметка не найдена: {note_id}", calls
            title = str(note.get("title", ""))
            content = str(note.get("content", ""))
            return f"{title}\n\n{content}".strip(), calls

        # search
        query = str(call.args["query"])
        k = int(call.args.get("k", 5))
        hits = self.tools.search(query, k=k)
        
        # если очень плохо — попробуем расширить запрос по ключевым словам
        if not hits or hits[0].score < 0.15:
            expanded = _expand_query(query)
            if expanded != query:
                calls.append(ToolCall(name="search", args={"query": expanded, "k": k}))
                hits2 = self.tools.search(expanded, k=k)
                if hits2:
                    hits = hits2

        return format_answer(user_input, hits), calls


def _find_md_token(text: str) -> Optional[str]:
    parts = text.replace("\\", "/").split()
    for p in parts:
        if p.lower().endswith(".md"):
            return p.strip("\"'(),. ")
    return None


def _expand_query(q: str) -> str:
    # убираем лишнюю пунктуацию и добавляем "obsidian" маркер
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in q)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return q
    if len(cleaned.split()) < 3:
        return f"{cleaned} заметка"
    return cleaned
