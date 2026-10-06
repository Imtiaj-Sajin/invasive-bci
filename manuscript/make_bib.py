"""Build refs.tex (Nature-style thebibliography) in order of first citation in main.tex.

Journal articles come from refs_meta.json (OpenAlex metadata, checked against Crossref on 2026-10-06); datasets,
conference papers, books and items without registered metadata are written out in MANUAL.

Usage: python manuscript/make_bib.py [--tex manuscript/main.tex] [--out manuscript/refs.tex]
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

ABBR = {
    "Nature": "Nature", "The Lancet": "Lancet", "eLife": "eLife", "New England Journal of Medicine": "N. Engl. J. Med.",
    "Nature Medicine": "Nat. Med.", "Journal of Neural Engineering": "J. Neural Eng.",
    "Science Translational Medicine": "Sci. Transl. Med.", "Nature Communications": "Nat. Commun.",
    "Nature Biomedical Engineering": "Nat. Biomed. Eng.", "Nature Neuroscience": "Nat. Neurosci.",
    "Communications Biology": "Commun. Biol.", "Neuron": "Neuron",
    "IEEE Transactions on Neural Systems and Rehabilitation Engineering": "IEEE Trans. Neural Syst. Rehabil. Eng.",
    "Neural Computation": "Neural Comput.", "ACS Chemical Neuroscience": "ACS Chem. Neurosci.",
    "Journal of Neuroscience Methods": "J. Neurosci. Methods", "Nature Methods": "Nat. Methods",
    "Frontiers in Bioengineering and Biotechnology": "Front. Bioeng. Biotechnol.",
    "Frontiers in Computational Neuroscience": "Front. Comput. Neurosci.",
}

# corrections from Crossref (2026-10-06) where OpenAlex differed
FIX = {
    "collinger2013": {"year": 2013, "last_page": "564"},
    "patel2023": {"year": 2023},
    "kozai2015": {"year": 2015},
    "ma2023": {"first_page": "e84296"},
    "pandarinath2017": {"first_page": "e18554"},
    "wilson2025": {"year": 2025, "last_page": "1484"},
    "jarosiewicz2015": {"first_page": "313ra179"},
    "card2026": {"last_page": "2510"},
}

MANUAL = {
    "sun2016": r"Sun, B., Feng, J. \& Saenko, K. Return of frustratingly easy domain adaptation. In \textit{Proc. 30th "
               r"AAAI Conference on Artificial Intelligence} 2058--2065 (AAAI Press, 2016).",
    "holm1979": r"Holm, S. A simple sequentially rejective multiple test procedure. \textit{Scand. J. Stat.} \textbf{6}, "
                r"65--70 (1979).",
    "loshchilov2019": r"Loshchilov, I. \& Hutter, F. Decoupled weight decay regularization. In \textit{Proc. 7th "
                      r"International Conference on Learning Representations} (2019).",
    "srivastava2014": r"Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. \& Salakhutdinov, R. Dropout: a simple "
                      r"way to prevent neural networks from overfitting. \textit{J. Mach. Learn. Res.} \textbf{15}, "
                      r"1929--1958 (2014).",
    "hahn2026": r"Hahn, N. V. et al. Performance of intracortical microelectrode arrays in people with implanted "
                r"brain--computer interfaces over a 20-year period. \textit{Nat. Med.} (in the press); preprint at "
                r"\url{https://doi.org/10.1101/2025.07.02.25330310} (2025).",
    "temmar2025": r"Temmar, H. et al. Long-term Intracortical Neural activity and Kinematics (LINK): an intracortical "
                  r"neural dataset for chronic brain--machine interfaces, neuroscience, and machine learning. "
                  r"\textit{Adv. Neural Inf. Process. Syst.} \textbf{38}, 154225--154247 (2025).",
    "bjanes2025": r"Bj{\aa}nes, D. A. et al. Quantifying physical degradation alongside recording and stimulation "
                  r"performance of 980 intracortical microelectrodes chronically implanted in three humans for "
                  r"956--2130 days. \textit{Acta Biomater.} \textbf{198}, 188--206 (2025).",
    "karpowicz2024falcon": r"Karpowicz, B. M. et al. Few-shot algorithms for consistent neural decoding (FALCON) "
                           r"benchmark. \textit{Adv. Neural Inf. Process. Syst.} \textbf{37} (2024).",
    "kuzborskij2013": r"Kuzborskij, I. \& Orabona, F. Stability and hypothesis transfer learning. In \textit{Proc. "
                      r"30th International Conference on Machine Learning} 942--950 (PMLR, 2013).",
    "sugiyama2007": r"Sugiyama, M., Krauledat, M. \& M{\"u}ller, K.-R. Covariate shift adaptation by importance "
                    r"weighted cross validation. \textit{J. Mach. Learn. Res.} \textbf{8}, 985--1005 (2007).",
    "patil2024": r"Patil, P., Du, J.-H. \& Tibshirani, R. J. Optimal ridge regularization for out-of-distribution "
                 r"prediction. In \textit{Proc. 41st International Conference on Machine Learning} (PMLR, 2024).",
    "hoerl1970": r"Hoerl, A. E. \& Kennard, R. W. Ridge regression: biased estimation for nonorthogonal problems. "
                 r"\textit{Technometrics} \textbf{12}, 55--67 (1970).",
    "hochreiter1997": r"Hochreiter, S. \& Schmidhuber, J. Long short-term memory. \textit{Neural Comput.} \textbf{9}, "
                      r"1735--1780 (1997).",
    "schonemann1966": r"Sch{\"o}nemann, P. H. A generalized solution of the orthogonal Procrustes problem. "
                      r"\textit{Psychometrika} \textbf{31}, 1--10 (1966).",
    "kingma2015": r"Kingma, D. P. \& Ba, J. Adam: a method for stochastic optimization. In \textit{Proc. 3rd "
                  r"International Conference on Learning Representations} (2015).",
    "liu1989": r"Liu, D. C. \& Nocedal, J. On the limited memory BFGS method for large scale optimization. "
               r"\textit{Math. Program.} \textbf{45}, 503--528 (1989).",
    "nelder1965": r"Nelder, J. A. \& Mead, R. A simplex method for function minimization. \textit{Comput. J.} "
                  r"\textbf{7}, 308--313 (1965).",
    "sobol1967": r"Sobol', I. M. On the distribution of points in a cube and the approximate evaluation of integrals. "
                 r"\textit{USSR Comput. Math. Math. Phys.} \textbf{7}, 86--112 (1967).",
    "field2007": r"Field, C. A. \& Welsh, A. H. Bootstrapping clustered data. \textit{J. R. Stat. Soc. B} "
                 r"\textbf{69}, 369--390 (2007).",
    "fisher1925": r"Fisher, R. A. \textit{Statistical Methods for Research Workers} (Oliver and Boyd, 1925).",
    "data_link": r"Temmar, H. et al. LINK: Long-Term Intracortical Neural Activity and Kinematics. \textit{DANDI "
                 r"Archive} \url{https://doi.org/10.48324/dandi.001201/0.251023.2336} (2025).",
    "data_perich": r"Perich, M. G., Miller, L. E., Azabou, M. \& Dyer, E. L. Long-term recordings of motor and "
                   r"premotor cortical spiking activity during reaching in monkeys. \textit{DANDI Archive} "
                   r"\url{https://doi.org/10.48324/dandi.000688/0.250122.1735} (2025).",
    "data_braingate": r"Hahn, N. V. et al. Data from: Performance of intracortical microelectrode arrays in people "
                      r"with implanted brain--computer interfaces over a 20-year period. \textit{Dryad} "
                      r"\url{https://doi.org/10.5061/dryad.x0k6djj1h} (2026).",
    "data_mindful": r"Pun, T. K. et al. Data from: Measuring instability in chronic human intracortical neural "
                    r"recordings towards stable, long-term brain--computer interfaces. \textit{Dryad} "
                    r"\url{https://doi.org/10.5061/dryad.n2z34tn5s} (2024).",
}


def initials(given):
    parts = re.split(r"[\s]+", given.replace(".", ". ").strip())
    out = []
    for p in parts:
        if not p:
            continue
        if "-" in p.strip("."):
            out.append("-".join(q[0] + "." for q in p.strip(".").split("-") if q))
        else:
            out.append(p[0] + ".")
    return " ".join(out)


def split_name(full):
    full = full.replace("\u2010", "-").replace("\u2011", "-")
    toks = full.split()
    particles = {"van", "von", "de", "der", "da", "di", "le"}
    i = len(toks) - 1
    while i > 0 and toks[i - 1].lower() in particles:
        i -= 1
    return " ".join(toks[i:]), " ".join(toks[:i])


def tex_escape(s):
    rep = {"\u2010": "-", "\u2011": "-", "\u2013": "--", "\u2014": "--", "\u2019": "'", "&": r"\&",
           "\u00e1": r"{\'a}", "\u00e9": r"{\'e}", "\u00f6": r'{\"o}', "\u00fc": r'{\"u}', "\u00e5": r"{\aa}",
           "\u00c1": r"{\'A}", "\u00f3": r"{\'o}", "\u00ed": r"{\'i}"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s


KEEP_CAPS = {"Utah", "Markov", "Procrustes", "BrainGate", "Bayesian", "Kalman", "LINK", "BMI", "FALCON"}


def sentence_case(t):
    """Nature style: sentence case. Only applied to titles written in Title Case (most words capitalized)."""
    words = t.split(" ")
    caps = [w for w in words[1:] if w[:1].isalpha()]
    if not caps or sum(w[0].isupper() for w in caps) / len(caps) < 0.6:
        return t
    def low(part):
        core = part.strip("():,.")
        if not core or core in KEEP_CAPS or (core.isupper() and len(core) > 1) or not core[:1].isupper():
            return part
        return part if any(c.isupper() for c in core[1:]) else part[:1].lower() + part[1:]

    first = words[0].split("-")
    out = ["-".join([first[0]] + [low(q) for q in first[1:]])]
    out += ["-".join(low(q) for q in w.split("-")) for w in words[1:]]
    return " ".join(out)


def title_case_fix(t):
    t = sentence_case(t)
    t = t.replace("brain-computer", "brain--computer").replace("Brain-Computer", "Brain--Computer")
    t = t.replace("brain\u2013computer", "brain--computer").replace("brain-machine", "brain--machine")
    t = t.replace("Brain-Machine", "Brain--Machine")
    return t[0].upper() + t[1:]


def format_article(key, m):
    m = {**m, **FIX.get(key, {})}
    names = [split_name(a) for a in m["authors"]]
    fmt = [f"{tex_escape(s)}, {tex_escape(initials(g))}" for s, g in names]
    if len(fmt) > 5:
        au = fmt[0] + " et al."
    elif len(fmt) == 1:
        au = fmt[0]
    else:
        au = ", ".join(fmt[:-1]) + r" \& " + fmt[-1]
    title = tex_escape(title_case_fix(m["title"].rstrip(".")))
    j = ABBR.get(m["venue"], m["venue"])
    pages = str(m.get("first_page") or "")
    if m.get("last_page") and m.get("last_page") != m.get("first_page") and pages.isdigit():
        pages += "--" + str(m["last_page"])
    vol = f" \\textbf{{{m['volume']}}}," if m.get("volume") else ""
    return f"{au} {title}. \\textit{{{j}}}{vol} {pages} ({m['year']}).".replace(" ,", ",").replace(",  (", " (")


def cite_order(tex):
    keys = []
    for grp in re.findall(r"\\cite\{([^}]*)\}", tex, flags=re.S):
        for k in grp.replace("\n", " ").split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(HERE, "main.tex"))
    ap.add_argument("--out", default=os.path.join(HERE, "refs.tex"))
    args = ap.parse_args()
    meta = json.load(open(os.path.join(HERE, "refs_meta.json"), encoding="utf-8"))
    keys = cite_order(open(args.tex, encoding="utf-8").read())
    lines = [r"\begin{thebibliography}{99}", ""]
    for k in keys:
        if k in MANUAL:
            entry = MANUAL[k]
        elif meta.get(k):
            entry = format_article(k, meta[k])
        else:
            raise SystemExit(f"no metadata for {k}")
        lines.append(f"\\bibitem{{{k}}} {entry}")
        lines.append("")
    lines.append(r"\end{thebibliography}")
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(keys)} references written to {args.out}")


if __name__ == "__main__":
    main()
