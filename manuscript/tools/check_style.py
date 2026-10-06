"""Style audit for the manuscript: word counts, long sentences, banned words and dash usage.

Usage: python manuscript/check_style.py [main.tex]
"""
import re
import sys

BANNED = ["novel", "unprecedented", "remarkabl", "striking", "extremely", "groundbreaking", "paradigm",
          "data not shown", "delve", "crucial", "pivotal", "landscape", "leverag", "utiliz", "furthermore",
          "moreover", "notably", "importantly", "interestingly"]


def strip(tex):
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\begin\{(figure|equation|itemize)\}.*?\\end\{\1\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\cite\{[^}]*\}", "", tex)
    tex = re.sub(r"\\(ref|label|url)\{[^}]*\}", "1", tex)
    tex = re.sub(r"\$[^$]*\$", "X", tex)
    tex = re.sub(r"\\(textit|textbf|emph|subsection\*|section\*|bmhead)\{([^}]*)\}", r"\2. ", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", tex)
    tex = tex.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"[ \t]+", " ", tex)


def section(tex, start, end):
    a = tex.index(start)
    b = tex.index(end, a)
    return tex[a:b]


def sentences(txt):
    txt = re.sub(r"\b(Fig|Figs|Supplementary|e\.g|i\.e|et al|ref|refs|d\.f|vs)\.", lambda m: m.group(0).replace(".", ""), txt)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", txt) if len(s.split()) > 2]


def main():
    tex = open(sys.argv[1] if len(sys.argv) > 1 else "manuscript/natcomms/main.tex", encoding="utf-8").read()
    abstract = strip(section(tex, "\\abstract{", "\\keywords"))
    title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n", tex, flags=re.S).group(1)
    main_txt = strip(section(tex, "\\section*{Introduction}", "\\section*{Methods}"))
    methods = strip(section(tex, "\\section*{Methods}", "\\section*{Data availability}"))
    legends = re.findall(r"\\caption\{(.*?)\}\n\\label", tex, flags=re.S)
    print(f"title words: {len(title.split())}  | abstract words: {len(abstract.split()) - 1}")
    print(f"main text words (intro+results+discussion): {len(main_txt.split())}")
    print(f"methods words: {len(methods.split())}")
    for i, lg in enumerate(legends, 1):
        print(f"legend {i}: {len(strip(lg).split())} words")
    for name, txt in (("abstract", abstract), ("main", main_txt), ("methods", methods)):
        ss = sentences(txt)
        lens = [len(s.split()) for s in ss]
        print(f"{name}: {len(ss)} sentences, mean {sum(lens) / len(lens):.1f} words, max {max(lens)}")
        for s, n in zip(ss, lens):
            if n > 32:
                print(f"   LONG ({n}): {s[:160]}")
    low = tex.lower()
    for w in BANNED:
        for m in re.finditer(w, low):
            print(f"banned '{w}': ...{tex[max(0, m.start() - 50):m.end() + 30]!r}")
    for m in re.finditer(r"---|\u2014", tex):
        print("EM DASH:", repr(tex[max(0, m.start() - 40):m.end() + 40]))
    subs = re.findall(r"\\subsection\*\{([^}]*)\}", section(tex, "\\section*{Results}", "\\section*{Discussion}"))
    for sname in subs:
        print(f"subheading {len(sname):2d} chars: {sname}")


if __name__ == "__main__":
    main()
