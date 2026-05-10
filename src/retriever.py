from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.schemas import FeatureGenerationRequest, RetrievalAudit, RetrievalItem


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "rag_Db.md"
TOKEN_PATTERN = re.compile(r"\b[\w-]{2,}\b", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    title: str
    content: str
    source: str


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS
    }


def _extract_section_title(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return "Untitled Section"
    first_line = lines[0]
    if first_line.startswith("## "):
        return first_line[3:].strip()
    return first_line.removeprefix("# ").strip()


def _parse_knowledge_base(path: Path) -> list[KnowledgeChunk]:
    raw_text = path.read_text(encoding="utf-8")
    sections = []
    for block in raw_text.split("\n---\n"):
        cleaned = block.strip()
        if not cleaned.startswith("## "):
            continue
        sections.append(
            KnowledgeChunk(
                title=_extract_section_title(cleaned),
                content=cleaned,
                source=path.name,
            )
        )
    if not sections:
        raise ValueError(f"No retrievable sections found in knowledge base: {path}")
    return sections


@lru_cache(maxsize=4)
def load_knowledge_chunks(path_str: str) -> tuple[KnowledgeChunk, ...]:
    return tuple(_parse_knowledge_base(Path(path_str)))


def build_retrieval_query(payload: FeatureGenerationRequest) -> str:
    column_descriptions = []
    for column in payload.columns:
        parts = [column.name, column.dtype]
        if column.description:
            parts.append(column.description)
        column_descriptions.append(" | ".join(parts))

    constraints = "; ".join(payload.constraints or [])
    return "\n".join(
        [
            f"Goal: {payload.project_goal}",
            f"Target: {payload.target_name} ({payload.target_type})",
            f"Dataset context: {payload.dataset_context or 'Not provided'}",
            f"Constraints: {constraints or 'None'}",
            "Columns:",
            *column_descriptions,
        ]
    )


def _score_chunk(
    chunk: KnowledgeChunk,
    query_tokens: set[str],
    dtype_tokens: set[str],
    target_type: str,
) -> float:
    title_tokens = _tokenize(chunk.title)
    content_tokens = _tokenize(chunk.content)
    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0

    score = float(len(overlap))
    score += 2.0 * len(query_tokens & title_tokens)

    lowered_title = chunk.title.lower()
    lowered_content = chunk.content.lower()

    if "avoid identity features" in lowered_title:
        score += 2.5
    if "target leakage" in lowered_title:
        score += 2.5
    if "match feature ideas to target type" in lowered_title and target_type in lowered_content:
        score += 2.0

    for dtype in dtype_tokens:
        if dtype in lowered_title or dtype in lowered_content:
            score += 1.5

    return round(score, 3)


def retrieve_knowledge(
    payload: FeatureGenerationRequest,
    *,
    knowledge_base_path: Path | None = None,
    top_k: int = 4,
) -> RetrievalAudit:
    kb_path = knowledge_base_path or DEFAULT_KNOWLEDGE_BASE_PATH
    chunks = load_knowledge_chunks(str(kb_path))
    query = build_retrieval_query(payload)
    query_tokens = _tokenize(query)
    dtype_tokens = {column.dtype.lower() for column in payload.columns}

    scored_items = []
    for chunk in chunks:
        score = _score_chunk(
            chunk,
            query_tokens=query_tokens,
            dtype_tokens=dtype_tokens,
            target_type=payload.target_type,
        )
        if score <= 0:
            continue
        scored_items.append((score, chunk))

    scored_items.sort(key=lambda item: item[0], reverse=True)
    selected = scored_items[:top_k]

    return RetrievalAudit(
        enabled=True,
        query=query,
        source=kb_path.name,
        items=[
            RetrievalItem(
                title=chunk.title,
                score=score,
                source=chunk.source,
            )
            for score, chunk in selected
        ],
    )


def format_retrieved_context(retrieval: RetrievalAudit) -> str:
    if not retrieval.items:
        return "No relevant knowledge base sections were retrieved."

    chunks = load_knowledge_chunks(str(DEFAULT_KNOWLEDGE_BASE_PATH))
    content_by_title = {chunk.title: chunk.content for chunk in chunks}
    formatted_sections = []
    for item in retrieval.items:
        content = content_by_title.get(item.title)
        if not content:
            continue
        formatted_sections.append(f"[Score: {item.score}] {content}")

    return "\n\n".join(formatted_sections) or "No relevant knowledge base sections were retrieved."
