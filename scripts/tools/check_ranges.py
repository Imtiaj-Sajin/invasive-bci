"""Find ranges in the manuscript written with individual-specific keys (e.g. \\val{T10:x}--\\val{T9:x} or
\\val{hum:x:min} to \\val{T7:x}) and check that the named individuals really are the extremes of that quantity.

Usage: python scripts/tools/check_ranges.py
"""
import re
import sys

sys.path.insert(0, "scripts")
D = "manuscript/natcomms"
IND = ["N", "C", "M", "T6", "T5", "T9", "T7", "T10", "T8"]
HUM = IND[3:]
PAT = re.compile(r"\\val\{([^}:]+):([^}]+)\}\s*(?:--|to)\s*\\val\{([^}:]+):([^}]+)\}")
DEF = re.compile(r"\\expandafter\\def\\csname val@(.+?)\\endcsname\{(.*)\}\s*$")


def num(s):
    s = str(s)
    if s == "$<$1":
        return 0.0
    try:
        return float(s.replace("$-$", "-").replace("\\%", ""))
    except ValueError:
        return None


def main():
    vals = {}
    for line in open(f"{D}/numbers.tex", encoding="utf-8"):
        m = DEF.match(line)
        if m:
            vals[m.group(1)] = m.group(2)
    bad = 0
    for f in ("main", "supplementary"):
        text = open(f"{D}/{f}.tex", encoding="utf-8").read()
        for m in PAT.finditer(text):
            a_ind, a_key, b_ind, b_key = m.groups()
            # quantity name without min/max suffix
            qa = re.sub(r":(min|max)$", "", a_key)
            qb = re.sub(r":(min|max)$", "", b_key)
            if qa != qb or (a_ind not in IND and a_ind not in ("hum", "all")) or (b_ind not in IND and b_ind not in ("hum", "all")):
                continue
            if a_ind in IND and b_ind in IND and a_ind == b_ind:
                continue                                       # same individual: e.g. its own min--max
            pool = HUM if (a_ind in HUM or b_ind in HUM or "hum" in (a_ind, b_ind)) else IND
            series = {i: num(vals.get(f"{i}:{qa}")) for i in pool}
            series = {i: v for i, v in series.items() if v is not None}
            if not series:
                continue
            lo, hi = min(series.values()), max(series.values())
            a_val, b_val = num(vals.get(f"{a_ind}:{a_key}")), num(vals.get(f"{b_ind}:{b_key}"))
            ok = a_val is not None and b_val is not None and abs(a_val - lo) < 1e-9 and abs(b_val - hi) < 1e-9
            if not ok:
                bad += 1
                print(f"{f}.tex: {m.group(0)}  -> {a_val} .. {b_val}, but range over {sorted(series)} is {lo} .. {hi}")
    print(f"{bad} suspicious ranges")


if __name__ == "__main__":
    main()
