from __future__ import annotations

import re


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def collapse_spaces(text: str) -> str:
    # 保留换行，压缩行内空格
    text = re.sub(r"[ \t]+", " ", text)
    return text


def collapse_blank_lines(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def remove_page_artifacts(text: str) -> str:
    # 先做保守清理，避免误删正文
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def clean_text(text: str) -> str:
    text = normalize_newlines(text)
    text = collapse_spaces(text)
    text = remove_page_artifacts(text)
    text = collapse_blank_lines(text)
    return text.strip()