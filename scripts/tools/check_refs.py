"""Check the manuscript's references: every citation has a bibitem, every bibitem is cited, and bibitems appear in
order of first citation (Nature Communications numbers references in citation order).

Usage: python scripts/tools/check_refs.py [--fix-order]   (--fix-order rewrites refs.tex in citation order)
"""
import re
import sys

D = "manuscript/natcomms"
CITE = re.compile(r"\\cite\{([^}]+)\}")
ITEM = re.compile(r"\\bibitem\{([^}]+)\}")


def main():
    main_tex = open(f"{D}/main.tex", encoding="utf-8").read()
    refs = open(f"{D}/refs.tex", encoding="utf-8").read()
    cites = []
    for grp in CITE.findall(main_tex):
        for k in (x.strip() for x in grp.split(",")):
            if k not in cites:
                cites.append(k)
    bib = ITEM.findall(refs)
    missing = [k for k in cites if k not in bib]
    unused = [k for k in bib if k not in cites]
    order_ok = [k for k in bib if k in cites] == [k for k in cites if k in bib]
    print(f"{len(cites)} cited keys, {len(bib)} bibitems; missing {missing}; never cited {unused}; "
          f"in citation order: {order_ok}")
    if "--fix-order" in sys.argv and not missing:
        head, rest = refs.split(r"\bibitem", 1)
        items = [it for it in re.split(r"(?=\\bibitem\{)", r"\bibitem" + rest) if it.strip()]
        tail = ""
        if r"\end{thebibliography}" in items[-1]:
            last, tail = items[-1].split(r"\end{thebibliography}", 1)
            items[-1] = last
            tail = r"\end{thebibliography}" + tail
        byk = {ITEM.match(it).group(1): it.rstrip() + "\n\n" for it in items}
        ordered = [byk[k] for k in cites] + [byk[k] for k in unused]
        open(f"{D}/refs.tex", "w", encoding="utf-8", newline="").write(head + "".join(ordered).rstrip() + "\n\n" + tail)
        print("refs.tex rewritten in citation order")


if __name__ == "__main__":
    main()
