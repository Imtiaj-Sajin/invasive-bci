"""Build the Journal of Neural Engineering version (manuscript/jne/main.tex) from the master text (natcomms/main.tex).

The master text is the single source. This script reuses its sections verbatim and changes only what JNE needs:
- IOP Publishing class (iopjournal.cls, in manuscript/jne), numbered sections, numeric references in brackets ordered by first citation (refs.tex rebuilt by
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

PREAMBLE = r"""\documentclass{iopjournal}

% IOP Publishing journal template (iopjournal.cls, 2024/01/31), downloaded from
% https://publishingsupport.iopscience.iop.org/questions/latex-template/ ; the class file is unchanged.
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{lineno}
\usepackage[numbers,sort&compress]{natbib}

% fill the template's placeholders (header and journal line) without editing the class
\fancyhead[L]{{\small \sf IOP Publishing}\hspace{5mm}{\it J. Neural Eng.} (submitted manuscript)}
\fancyhead[R]{M I A Sajin {\it et al}\ }
\renewcommand{\articletype}[1]{{\vspace*{-8mm}\noindent\Large\sf Journal of Neural Engineering}\par
  \vspace*{6mm}\noindent{\scriptsize\sf\bfseries\MakeUppercase{#1}}\par\vspace*{2mm}}

\raggedbottom
\graphicspath{{../../results/figures/v2/}}
\input{../natcomms/numbers}

\begin{document}

\articletype{Paper}

\title{%TITLE%}

\author{Md. Imtiaj Alam Sajin$^{1,*}$\orcid{0009-0009-2423-1835}, Esm E Moula Chowdhury
Abha$^{1}$\orcid{0009-0008-3776-2283} and Md Wahiduzzaman Suva$^{1}$\orcid{0009-0007-6227-7282}}

\affil{$^1$Department of Computer Science, American International University-Bangladesh, Dhaka 1229, Bangladesh}

\affil{$^*$Author to whom any correspondence should be addressed.}

\email{imtiajsajin@gmail.com}

\keywords{%KEYWORDS%}

\begin{abstract}
\input{abstract}
\end{abstract}

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

    # Methods come first in JNE: the short overview that defines the terms and cites Fig. 2 opens the Methods
    ov_start = "\\subsection*{Datasets and approach}"
    a = results.index(ov_start)
    b = results.index("\\subsection*{", a + len(ov_start))
    overview, results = results[a + len(ov_start):b].strip("\n"), results[:a] + results[b:]
    methods = "\\subsection*{Overview of the approach}\n" + overview + "\n\n" + methods.lstrip("\n")

    funding = "This work received no specific funding."
    ack = ack.replace(funding, "").strip()
    parts = [
        PREAMBLE.replace("%TITLE%", title).replace("%KEYWORDS%", keywords),
        "\\section{Introduction}\n" + intro,
        "\\section{Methods}\n" + numbered(methods),
        "\\section{Results}\n" + numbered(results),
        "\\section{Discussion}\n" + discussion,
        "\\section*{Use of AI tools}\n" + ai,
        "\\ack{" + ack + "}",
        "\\funding{" + funding + "}",
        "\\roles{" + contrib + "}",
        "\\data{" + data + "\n\n" + code + "}",
        "\\section*{Conflict of interest}\n" + compete,
        "\\input{refs}\n\n\\end{document}\n",
    ]
    out = "\n\n".join(p.strip("\n") for p in parts) + "\n"
    out = out.replace("Supplementary Information", "Supplementary Material")
    out = out.replace("width=170mm", "width=\\textwidth")             # IOP text width is 153 mm
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
