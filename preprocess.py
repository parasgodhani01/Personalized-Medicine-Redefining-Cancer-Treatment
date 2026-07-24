# preprocess.py
# ─────────────────────────────────────────────────────────────
# Shared text cleaning used by BOTH train.py and main.py.
# CRITICAL: this must produce IDENTICAL output at train time and
# inference time, or the model will see features it wasn't trained on.
# ─────────────────────────────────────────────────────────────

import re

# Small fixed stopword list — no external download (nltk requires a
# runtime download step that breaks in CI/offline environments).
STOPWORDS = {
    "the", "is", "a", "an", "in", "on", "at", "of", "to", "for",
    "and", "or", "but", "with", "as", "by", "from", "that", "this",
    "it", "be", "are", "was", "were", "been", "has", "have", "had",
    "we", "our", "which", "these", "those", "into", "such"
}


def clean_text(text) -> str:
    """
    Lowercase, strip numbers/special chars, remove stopwords.
    Must handle None and empty string without crashing.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # drop digits + punctuation/special chars
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


def build_combined_feature(gene: str, variation: str, clinical_text: str) -> str:
    """
    Combine gene + variation + cleaned clinical text into one string
    for TF-IDF. Gene/variation are lowercased but NOT passed through
    clean_text (which strips digits — would destroy variation IDs like R1699Q).
    """
    gene_part = (gene or "").lower().strip()
    var_part  = (variation or "").lower().strip()
    text_part = clean_text(clinical_text)

    return f"{gene_part} {var_part} {text_part}".strip()