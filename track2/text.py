"""The organisers' transcript normaliser (tags removed, punctuation to spaces, lower case)."""
import re
import unicodedata

TAG_RE = re.compile(r"</?[^<>]*>|\[[^\[\]]*\]")


def normalize(text):
    s = TAG_RE.sub(" ", text or "")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.lower().split())
