"""Build the Journal of Neural Engineering version (manuscript/jne/main.tex) from the master text (natcomms/main.tex).

The master text is the single source. This script reuses its sections verbatim and changes only what JNE needs:
- article class, numbered sections, numeric references in brackets ordered by first citation (refs.tex rebuilt by
  make_bib.py for the new order);
- structured abstract from jne/abstract.tex (Objective, Approach, Main results, Significance);
- section order Introduction, Methods, Results, Discussion (IOP convention), then the declarations.
Numbers (\\val) and figures come from the same files as the master text, so the versions cannot disagree.

Usage: python manuscript/tools/make_jne.py   (then compile manuscript/jne/main.tex with pdflatex twice)
"""
import io
import os
import re

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "natcomms", "main.tex")
OUT_DIR = os.path.join(HERE, "..", "jne")

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[margin=20mm]{geometry}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{lineno}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period}

\raggedbottom
\graphicspath{{../../results/figures/v2/}}
\input{../natcomms/numbers}

\title{%TITLE%}
\author{Md. Imtiaj Alam Sajin$^{1,*}$, Esm E Moula Chowdhury Abha$^{1}$ and Md Wahiduzzaman Suva$^{1}$\\[4pt]
\small $^{1}$Department of Computer Science, American International University-Bangladesh, Dhaka 1229, Bangladesh\\
\small $^{*}$Author for correspondence: imtiajsajin@gmail.com\\
\small ORCID: Sajin 0009-0009-2423-1835; Chowdhury Abha 0009-0008-3776-2283; Suva 0009-0007-6227-7282}
\date{}

\begin{document}
\maketitle

\begin{abstract}
\input{abstract}
\end{abstract}

\noindent\textbf{Keywords:} %KEYWORDS%

\linenumbers
"""


def block(s, start, ends):
    """Text from marker `start` (exclusive) to the first of `ends` that follows it."""
    a = s.index(start) + len(start)
    b = min(s.index(e, a) for e in ends if e in s[a:])
    return s[a:b].strip("\n")


def main():
    s = io.open(SRC, encoding="utf-8").read()
    title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n", s, re.S).group(1).replace("\n", " ")
    keywords = re.search(r"\\keywords\{(.*?)\}\n", s, re.S).group(1).replace("\n", " ")
    intro = block(s, "\\section*{Introduction}", ["\\section*{Results}"])
    results = block(s, "\\section*{Results}", ["\\section*{Discussion}"])
    discussion = block(s, "\\section*{Discussion}", ["\\section*{Methods}"])
    methods = block(s, "\\section*{Methods}", ["\\subsection*{Use of AI tools}"])
    ai = block(s, "\\subsection*{Use of AI tools}", ["\\section*{Data availability}"])
    data = block(s, "\\section*{Data availability}", ["\\section*{Code availability}"])
    code = block(s, "\\section*{Code availability}", ["\\backmatter"])
    ack = block(s, "\\bmhead{Acknowledgements}", ["\\bmhead{"])
    contrib = block(s, "\\bmhead{Author contributions}", ["\\bmhead{"])
    compete = block(s, "\\bmhead{Competing interests}", ["\\input{refs}"])

    def numbered(t):
        return t.replace("\\subsection*{", "\\subsection{")

    parts = [
        PREAMBLE.replace("%TITLE%", title).replace("%KEYWORDS%", keywords),
        "\\section{Introduction}\n" + intro,
        "\\section{Methods}\n" + numbered(methods),
        "\\section{Results}\n" + numbered(results),
        "\\section{Discussion}\n" + discussion,
        "\\section*{Data availability}\n" + data,
        "\\section*{Code availability}\n" + code,
        "\\section*{Use of AI tools}\n" + ai,
        "\\section*{Acknowledgements}\n" + ack,
        "\\section*{Author contributions}\n" + contrib,
        "\\section*{Conflict of interest}\n" + compete,
        "\\input{refs}\n\n\\end{document}\n",
    ]
    out = "\n\n".join(p.strip("\n") for p in parts) + "\n"
    out = out.replace("Supplementary Information", "Supplementary Material")
    out = re.sub(r"(?<=[^\s~{(])\\cite\{", "~\\\\cite{", out)      # bracketed numbers need a space before them
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "main.tex")
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    print("wrote", os.path.normpath(path))
    # references: the master's verified entries, reordered by first citation in this version
    sys.path.insert(0, HERE)
    from make_bib import cite_order
    refs = io.open(os.path.join(HERE, "..", "natcomms", "refs.tex"), encoding="utf-8").read()
    entries = dict(re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\n\s*\n|\n\\end\{thebibliography\})", refs, re.S))
    keys = cite_order(out)
    missing = [k for k in keys if k not in entries]
    if missing:
        raise SystemExit(f"cited but not in natcomms/refs.tex: {missing}")
    lines = ["\\begin{thebibliography}{99}", ""]
    for k in keys:
        lines += [f"\\bibitem{{{k}}} {entries[k]}", ""]
    lines.append("\\end{thebibliography}")
    io.open(os.path.join(OUT_DIR, "refs.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"{len(keys)} references (of {len(entries)} in the master) written in citation order")


if __name__ == "__main__":
    main()
