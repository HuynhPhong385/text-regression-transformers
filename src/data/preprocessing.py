"""Basic text preprocessing."""

import re


def clean_text(text):
    """Chuẩn hóa whitespace, giữ nguyên nội dung/ngữ nghĩa."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
