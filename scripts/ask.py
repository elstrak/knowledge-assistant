import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.agent import AgentLoop, Tools
from ka.rag import Retriever


def main() -> None:
    def safe_write(s: str) -> None:
        # Write UTF-8 bytes (Cursor terminal/capture expects UTF-8); replace unsupported chars safely.
        sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))

    parser = argparse.ArgumentParser(description="Ask Knowledge Assistant (RAG baseline)")
    parser.add_argument(
        "--index",
        default="dataset/index",
        help="Index directory (default: dataset/index)",
    )
    parser.add_argument(
        "--notes",
        default="dataset/processed/notes.jsonl",
        help="Notes jsonl path (default: dataset/processed/notes.jsonl)",
    )
    parser.add_argument("--q", required=True, help="User question")
    args = parser.parse_args()

    q = _maybe_fix_mojibake(args.q)

    index_dir = os.path.abspath(os.path.expanduser(args.index))
    notes_path = os.path.abspath(os.path.expanduser(args.notes))

    retriever = Retriever(index_dir=index_dir)
    tools = Tools(retriever=retriever, notes_path=notes_path)
    agent = AgentLoop(tools=tools)

    answer, calls = agent.run(q)
    safe_write(answer + "\n")
    safe_write("\n---\nTool calls:\n")
    for c in calls:
        safe_write(f"- {c.name}: {c.args}\n")


def _maybe_fix_mojibake(s: str) -> str:
    """
    Heuristic for Windows terminals: sometimes UTF-8 bytes get decoded as cp1251,
    producing strings like "РјРѕСЂ...". Try to recover.
    """
    if not s:
        return s
    # typical mojibake markers for Cyrillic in cp1251-decoded UTF-8
    if "Р" not in s and "С" not in s:
        return s
    try:
        fixed = s.encode("cp1251", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return s
    # accept if it looks like real Cyrillic text
    cyr = sum(1 for ch in fixed if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    return fixed if cyr >= 2 else s


if __name__ == "__main__":
    main()


