import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise SystemExit(
        "Этот скрипт требует PyYAML. Установи пакет командой: pip install pyyaml"
    ) from e


# --- Регулярки --------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w/-]+)", re.UNICODE)


@dataclass
class Note:
    id: str
    title: str
    tags: List[str]
    links: List[str]
    content: str
    created: Optional[str] = None
    modified: Optional[str] = None


def split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    fm_text = m.group(1)
    body = text[m.end():]

    try:
        data = yaml.safe_load(fm_text) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}
        body = text

    return data, body


def extract_title(frontmatter: Dict[str, Any], body: str, fallback_filename: str) -> str:
    for key in ("title", "Title"):
        if key in frontmatter and isinstance(frontmatter[key], str):
            t = frontmatter[key].strip()
            if t:
                return t

    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("#"):
            cleaned = line.lstrip("#").strip()
            if cleaned:
                return cleaned

    return os.path.splitext(os.path.basename(fallback_filename))[0]


def normalize_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []

    if isinstance(raw_tags, str):
        tags = [raw_tags]
    elif isinstance(raw_tags, list):
        tags = [str(t) for t in raw_tags]
    else:
        return []

    cleaned = []
    for t in tags:
        t = t.strip()
        if t.startswith("#"):
            t = t[1:]
        if t:
            cleaned.append(t)
    return cleaned


def extract_tags(frontmatter: Dict[str, Any], body: str) -> List[str]:
    fm_tags: List[str] = []

    for key in ("tags", "tag"):
        if key in frontmatter:
            fm_tags = normalize_tags(frontmatter[key])
            break

    inline_tags = TAG_RE.findall(body)
    all_tags = list({*fm_tags, *inline_tags})
    return all_tags


def extract_links(text: str) -> List[str]:
    links: List[str] = []
    for match in WIKILINK_RE.findall(text):
        target = match.split("|", 1)[0].strip()
        if target:
            links.append(target)
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    return unique_links


def iso_timestamp_from_stat(path: str, attr: str) -> Optional[str]:
    try:
        st = os.stat(path)
        ts = getattr(st, attr, None)
        if ts is None:
            return None
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except OSError:
        return None


def parse_markdown_file(path: str, vault_root: str) -> Note:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    frontmatter, body = split_frontmatter(text)

    rel_path = os.path.relpath(path, vault_root).replace(os.sep, "/")
    title = extract_title(frontmatter, body, fallback_filename=path)
    tags = extract_tags(frontmatter, body)
    links = extract_links(text)

    created = iso_timestamp_from_stat(path, "st_ctime")
    modified = iso_timestamp_from_stat(path, "st_mtime")

    return Note(
        id=rel_path,
        title=title,
        tags=tags,
        links=links,
        content=body,
        created=created,
        modified=modified,
    )


def iter_markdown_files(vault_path: str, exclude_dirs: List[str]) -> List[str]:
    md_files: List[str] = []

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            full_path = os.path.join(root, name)
            md_files.append(full_path)

    return md_files


def main() -> None:
    print("[DEBUG] collect_obsidian стартовал")

    parser = argparse.ArgumentParser(
        description="Собирает заметки Obsidian в единый JSONL-файл."
    )
    parser.add_argument(
        "--vault-path",
        required=True,
        help="Путь к корневой папке Obsidian vault",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Путь к выходному JSONL-файлу (например, processed/notes.jsonl)",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Имя директории, которую нужно исключить (можно указать несколько раз).",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Тег (без #), при наличии которого заметка будет исключена.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Печатать отладочную информацию.",
    )

    args = parser.parse_args()
    print(f"[DEBUG] Параметры: {args}")

    vault_path = os.path.abspath(os.path.expanduser(args.vault_path))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    exclude_dirs = args.exclude_dir
    exclude_tags = set(args.exclude_tag)
    verbose = args.verbose

    if not os.path.isdir(vault_path):
        raise SystemExit(f"[ERROR] Vault path не существует или не является директорией: {vault_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Сканирую vault: {vault_path}")
    print(f"[INFO] Исключаем директории: {exclude_dirs}")
    if exclude_tags:
        print(f"[INFO] Исключаем заметки с тегами: {sorted(exclude_tags)}")

    md_files = iter_markdown_files(vault_path, exclude_dirs)
    print(f"[INFO] Найдено markdown-файлов: {len(md_files)}")

    if len(md_files) == 0:
        raise SystemExit(
            "[ERROR] Не найдено ни одного .md файла"
        )

    count_total = 0
    count_written = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in md_files:
            count_total += 1
            if verbose and count_total <= 5:
                print(f"[DEBUG] Обрабатываю: {path}")

            note = parse_markdown_file(path, vault_root=vault_path)

            if exclude_tags and any(t in exclude_tags for t in note.tags):
                if verbose:
                    print(f"[DEBUG] Пропускаю {path} по exclude_tag, теги: {note.tags}")
                continue

            out_f.write(json.dumps(asdict(note), ensure_ascii=False) + "\n")
            count_written += 1

    print(f"[INFO] Обработано файлов всего: {count_total}")
    print(f"[INFO] Записано заметок в {output_path}: {count_written}")


if __name__ == "__main__":
    main()
