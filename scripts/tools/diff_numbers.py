"""List generated manuscript numbers that changed between two numbers.tex files, marking those used in the text.

Usage: python scripts/tools/diff_numbers.py OLD_numbers.tex [NEW_numbers.tex] [--all]
"""
import re
import sys

PAT = re.compile(r"\\expandafter\\def\\csname val@(.+?)\\endcsname\{(.*)\}\s*$")


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        m = PAT.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    old = load(args[0])
    new = load(args[1] if len(args) > 1 else "manuscript/natcomms/numbers.tex")
    text = "".join(open(f"manuscript/natcomms/{f}.tex", encoding="utf-8").read() for f in ("main", "supplementary"))
    used = set(re.findall(r"\\val\{([^}]+)\}", text))
    changed = sorted(k for k in old if k in new and old[k] != new[k])
    print(f"{len(old)} old keys, {len(new)} new keys; {len(changed)} changed, "
          f"{sum(k in used for k in changed)} of them used in the text")
    print("used in text but missing now:", sorted(k for k in used if k not in new))
    for k in changed:
        if k in used or "--all" in sys.argv:
            print(f"  {k}: {old[k]}  ->  {new[k]}")


if __name__ == "__main__":
    main()
