#!/usr/bin/env python

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

print("[DEBUG] preprocess_obsidian.py: файл импортирован, код выполняется.")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    note_id: str
    title: str
    section: str
    text: str
    tags: List[str]
    links: List[str]
    position: int


def split_into_sections(content: str, note_title: str) -> List[Tuple[str, str]]:
    lines = content.splitlines()

    sections: List[Tuple[str, List[str]]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_title is not None or current_lines:
                title_for_section = current_title if current_title is not None else "Introduction"
                sections.append((title_for_section, current_lines))
                current_lines = []
            heading_text = m.group(2).strip()
            current_title = heading_text if heading_text else "Section"
        else:
            current_lines.append(line)

    if current_title is not None or current_lines:
        title_for_section = current_title if current_title is not None else note_title
        sections.append((title_for_section, current_lines))

    if not sections:
        return []

    result: List[Tuple[str, str]] = []
    for sec_title, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if text:
            result.append((sec_title, text))

    if not result:
        return [(note_title, "")]

    return result


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    n = len(words)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
    # продолжение цикла
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def main() -> None:
    print("[DEBUG] main() стартовал")

    input_path = os.path.abspath("processed/notes.jsonl")
    output_path = os.path.abspath("processed/chunks.jsonl")

    print(f"[DEBUG] Ожидаемый вход: {input_path}")
    print(f"[DEBUG] Ожидаемый выход: {output_path}")
    print(f"[DEBUG] Текущая директория: {os.getcwd()}")

    if not os.path.isfile(input_path):
        print("[ERROR] Файл processed/notes.jsonl не найден.")
        return

    if os.path.getsize(input_path) == 0:
        print("[ERROR] processed/notes.jsonl пустой.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chunk_size = 800
    overlap = 200

    count_notes = 0
    count_chunks = 0

    with open(input_path, "r", encoding="utf-8") as in_f, open(
        output_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            note = json.loads(line)

            note_id = note.get("id", "")
            note_title = note.get("title", "") or os.path.basename(note_id)
            tags = note.get("tags", []) or []
            links = note.get("links", []) or []
            content = note.get("content", "") or ""

            count_notes += 1
            if count_notes <= 3:
                print(f"[DEBUG] Обрабатываю заметку #{count_notes}: {note_id} (title={note_title})")

            chunk_index = 0

            sections = split_into_sections(content, note_title)
            if count_notes <= 3:
                print(f"[DEBUG]   Секций в заметке: {len(sections)}")

            for section_title, section_text in sections:
                raw_chunks = chunk_text(section_text, chunk_size, overlap)
                for text_chunk in raw_chunks:
                    text_chunk = text_chunk.strip()
                    if not text_chunk:
                        continue
                    chunk_index += 1
                    count_chunks += 1

                    if count_chunks <= 5:
                        print(f"[DEBUG]   Создаю чанк #{chunk_index} для {note_id} "
                              f"(section={section_title[:30]!r})")

                    chunk = Chunk(
                        chunk_id=f"{note_id}#{chunk_index}",
                        note_id=note_id,
                        title=note_title,
                        section=section_title,
                        text=text_chunk,
                        tags=tags,
                        links=links,
                        position=chunk_index,
                    )
                    out_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"[INFO] Обработано заметок: {count_notes}")
    print(f"[INFO] Сгенерировано чанков: {count_chunks}")
    print(f"[INFO] Результат записан в: {output_path}")


if __name__ == "__main__":
    print("[DEBUG] __main__ ветка — вызываем main()")
    main()
