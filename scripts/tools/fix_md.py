"""Tidy markdown spacing: blank lines around headings, lists, tables and fences; spaced table separators.

Usage: python scripts/tools/fix_md.py FILE [FILE ...]
"""
import re
import sys


def fix(text):
    lines = text.split("\n")
    out = []
    in_fence = False
    for l in lines:
        is_fence = l.startswith("```")
        kind = ("fence" if is_fence else "head" if l.startswith("#") else "table" if l.startswith("|")
                else "list" if re.match(r"^(\s*[-*] |\s*\d+\. )", l) else "text" if l.strip() else "blank")
        prev = out[-1] if out else ""
        prev_kind = ("head" if prev.startswith("#") else "table" if prev.startswith("|")
                     else "list" if re.match(r"^(\s*[-*] |\s*\d+\. )", prev) or (prev.startswith("  ") and prev.strip())
                     else "text" if prev.strip() else "blank")
        if not in_fence and out and prev_kind != "blank":
            if kind in ("head", "fence") or prev_kind == "head" or (kind != prev_kind and "table" in (kind, prev_kind)) \
                    or (kind == "list" and prev_kind == "text"):
                if not (is_fence and in_fence):
                    out.append("")
        if kind == "table" and re.match(r"^\|[-|: ]+\|$", l):
            l = "| " + " | ".join("---" for _ in l.strip("|").split("|")) + " |"
        out.append(l)
        if is_fence:
            in_fence = not in_fence
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path, encoding="utf8") as f:
            t = f.read()
        with open(path, "w", encoding="utf8") as f:
            f.write(fix(t))
