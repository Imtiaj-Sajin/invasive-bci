"""Summarize the NDT3 pilot against the linear decoder on the same threshold-crossing pairs.

Inputs: results/ndt3/pilot/<subject><tag>.csv (final runs: N and T5 at the selected learning rate, T8 with the array
rule) and DATA/ndt3_pilot/pairs.csv (linear reference methods on the same pairs, all channels).
Output: results/ndt3/pilot_summary.json, used by make_numbers_tex.py and make_supp_tables.py.

Comparisons (medians over pairs; retention = R2 / the model's own same-day R2 on the test day):
  same day           NDT3 reference vs linear reference (per test session)
  no update          NDT3 zero-shot vs renormalized linear decoder
  50 / 300 trials    NDT3 few-shot vs linear refit toward the previous decoder at the same budget
  unsupervised       NDT3 with the neural loss only (no linear counterpart)
Usage: python scripts/ndt3/summarize_pilot.py
"""
import json
import os

import pandas as pd

COLS = ["subject", "kind", "train", "test", "gap_target", "condition", "r2", "epochs", "val", "bins", "seconds",
        "base_epochs", "base_val"]
RUNS = {"N": "_lr4e-4", "T5": "_lr4e-4", "T8": "_rule"}
PAIRS = "results/ndt3/pilot_pairs_linear.csv"   # copy of <pilot data>/pairs.csv written by export_pilot.py
OUT = "results/ndt3/pilot_summary.json"
COMPARE = {"zeroshot": "renorm", "fewshot_n50": "prior_refit_n50", "fewshot_n300": "prior_refit_n300"}


def main():
    pairs = pd.read_csv(PAIRS)
    out = {}
    for k, tag in RUNS.items():
        path = f"results/ndt3/pilot/{k}{tag}.csv"
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, header=None, skiprows=1, names=COLS)
        p = pairs[pairs.subject == k]
        ref = d[d.kind == "ref"].set_index("test").r2
        q = d[d.kind == "pair"].pivot_table(index=["train", "test", "gap_target"], columns="condition",
                                            values="r2").reset_index()
        q["ndt3_ref"] = q.test.map(ref)
        q = q.merge(p, on=["train", "test", "gap_target"], how="left")
        sd = q.drop_duplicates("test")
        r = {"run": tag, "n_pairs": int(len(q)), "n_sessions": int(sd.test.nunique()),
             "sameday": {"ndt3": float(sd.ndt3_ref.median()), "linear": float(sd.reference.median()),
                         "ndt3_better": int((sd.ndt3_ref > sd.reference).sum())}}
        for c, lin in COMPARE.items():
            if c not in q:
                continue
            v = q[[c, lin, "ndt3_ref", "reference"]].dropna()
            r[c] = {"ndt3": float(v[c].median()), "linear": float(v[lin].median()),
                    "ndt3_ret": float((v[c] / v.ndt3_ref).median()), "linear_ret": float((v[lin] / v.reference).median()),
                    "ndt3_better": int((v[c] > v[lin]).sum()), "n": int(len(v))}
        if "unsup" in q:
            v = q[["unsup", "zeroshot", "ndt3_ref"]].dropna()
            r["unsup"] = {"ndt3": float(v.unsup.median()), "ndt3_ret": float((v.unsup / v.ndt3_ref).median()),
                          "better_than_zeroshot": int((v.unsup > v.zeroshot).sum()), "n": int(len(v))}
        r["epoch_cap_hits"] = int((d.epochs >= 99).sum())
        r["rows"] = int(len(d))
        out[k] = r
        print(k, json.dumps(r, indent=None)[:400], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
