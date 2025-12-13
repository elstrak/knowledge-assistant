import json
import os
import re
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def clean_markdown(text: str) -> str:
    """
    Лёгкая очистка markdown, чтобы предложения читались ровнее.
    """
    # заголовки "# ", "## " и т.п.
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    # маркеры списков "* ", "- ", "1. " и т.д.
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # лишние подчёркивания/разделители
    text = re.sub(r'[_*`]{2,}', ' ', text)
    # несколько пробелов/переносов → один пробел
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

def split_into_sentences(text: str) -> List[str]:
    """
    Очень простой sentence splitter
    Берём всё, что заканчивается на . ? ! … и т.п., плюс остаток.
    """
    text = text.strip()
    if not text:
        return []

    text = re.sub(r'\s+', ' ', text)

    sentence_end_re = re.compile(r'(.+?[.!?…]+)(\s+|$)')
    sentences = []
    last_end = 0

    for m in sentence_end_re.finditer(text):
        sent = m.group(1).strip()
        if sent:
            sentences.append(sent)
        last_end = m.end()

    tail = text[last_end:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Чанкует текст по предложениям
    """
    text = clean_markdown(text)
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    def count_words(s: str) -> int:
        return len(s.split())

    for sent in sentences:
        sent_tokens = count_words(sent)

        # Если предложение само по себе больше chunk_size — положим его отдельным чанком
        # (иначе застрянем в бесконечном разбиении).
        if sent_tokens >= chunk_size:
            # сначала закрываем текущий чанк, если он есть
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            chunks.append(sent)
            continue

        # Попробуем добавить предложение в текущий чанк
        if current_tokens + sent_tokens <= chunk_size:
            current_sentences.append(sent)
            current_tokens += sent_tokens
        else:
            # Текущий чанк заполнен → закрываем его
            if current_sentences:
                chunks.append(" ".join(current_sentences))

                # Делаем overlap по словам: берём последние overlap слов
                if overlap > 0:
                    merged = " ".join(current_sentences)
                    words = merged.split()
                    if len(words) > overlap:
                        overlap_words = words[-overlap:]
                    else:
                        overlap_words = words  # чанк и так маленький
                    overlap_text = " ".join(overlap_words)
                    current_sentences = [overlap_text, sent]
                    current_tokens = count_words(overlap_text) + sent_tokens
                else:
                    current_sentences = [sent]
                    current_tokens = sent_tokens
            else:
                # если почему-то текущий пуст (крайний случай) — просто начинаем новый
                current_sentences = [sent]
                current_tokens = sent_tokens

    # добиваем последний чанк
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Чанкует собранные заметки Obsidian (notes.jsonl) в chunks.jsonl."
    )
    parser.add_argument(
        "--input",
        default="dataset/processed/notes.jsonl",
        help="Путь к входному JSONL с заметками (по умолчанию: dataset/processed/notes.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="dataset/processed/chunks.jsonl",
        help="Путь к выходному JSONL с чанками (по умолчанию: dataset/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Размер чанка в словах (примерно). По умолчанию: 300",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=60,
        help="Overlap в словах между чанками. По умолчанию: 60",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Печатать отладочную информацию (первые N заметок/чанков).",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    chunk_size = args.chunk_size
    overlap = args.overlap
    verbose = args.verbose

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] chunk_size={chunk_size}, overlap={overlap}")

    if not os.path.isfile(input_path):
        raise SystemExit(f"[ERROR] Input файл не найден: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise SystemExit(f"[ERROR] Input файл пустой: {input_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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
            if verbose and count_notes <= 3:
                print(f"[DEBUG] Обрабатываю заметку #{count_notes}: {note_id} (title={note_title})")

            chunk_index = 0

            sections = split_into_sections(content, note_title)
            if verbose and count_notes <= 3:
                print(f"[DEBUG]   Секций в заметке: {len(sections)}")

            for section_title, section_text in sections:
                raw_chunks = chunk_text(section_text, chunk_size, overlap)
                for text_chunk in raw_chunks:
                    text_chunk = text_chunk.strip()
                    if not text_chunk:
                        continue
                    chunk_index += 1
                    count_chunks += 1

                    if verbose and count_chunks <= 5:
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
    main()
