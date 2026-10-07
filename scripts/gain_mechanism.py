"""Do fitted per-channel weights track measured changes at each electrode? (BrainGate participants)

For each pair (day i -> day j; decoders with alpha = 1e4), the re-weighting correction is fitted with day j's first 300
labelled trials (signed weights, strength by cross-validation, as in the ladder). The fitted weight of each channel is
compared with independent per-electrode measurements from the yield recordings of the same two days:
  d_rate   change in log threshold-crossing rate at -4.5 RMS, log(rate_j + 0.1) - log(rate_i + 0.1)
  d_imp    change in log impedance (values >= 4,000 kOhm excluded)
For each pair we compute the Spearman correlation between weights and each change across channels, and summarize the
correlations over training sessions (median, share positive, two-sided Wilcoxon signed-rank test against zero).
A null correlation is obtained by permuting channels.

Usage: python scripts/gain_mechanism.py --subject T6 [--max-train 40] [--out results/gain_mechanism]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from decay_alpha import GAPS, sessions_for  # noqa: E402
from ibci.anatomy import LAMS_CF, cv_shrunk, gain_features  # noqa: E402


def yield_day(root, p, day):
    f = glob.glob(os.path.join(root, "yield", p, f"{p}_day_{day}_yield.mat"))
    if not f:
        return None
    d = loadmat(f[0], simplify_cells=True)
    el = pd.DataFrame(d["electrodes"])
    fr = pd.DataFrame(d["firing_rates"])
    fr = fr[fr.threshold_description.astype(str).isin(["-4.5", "-4.50"])]
    el["rate"] = el.electrode_id.map(fr.set_index("electrode_id").firing_rate)
    if "impedance" not in el:
        el["impedance"] = np.nan
    el["impedance"] = el.impedance.where(el.impedance < 4000)
    return el.set_index("electrode_id")[["rate", "impedance"]]


def fit_gains(dec, sj, n=300):
    Ftr = gain_features(sj.ztr, dec)
    tid = np.asarray(sj.trial_id)
    lab = tid < n
    lam, _ = cv_shrunk(Ftr, sj.y_tr, tid, n, LAMS_CF, np.ones(Ftr.shape[1]), 5, intercept_anchor=dec.b)
    T, C, K = Ftr[lab].shape
    A = np.concatenate([Ftr[lab].transpose(0, 2, 1).reshape(T * K, C), np.tile(np.eye(K), (T, 1))], axis=1)
    s = np.sqrt(lam * T)
    A_aug = np.vstack([A, s * np.eye(C + K)])
    y_aug = np.concatenate([sj.y_tr[lab].reshape(-1), s * np.r_[np.ones(C), np.asarray(dec.b, float)]])
    return np.linalg.lstsq(A_aug, y_aug, rcond=None)[0][:C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--max-train", type=int, default=40)
    ap.add_argument("--out", default="results/gain_mechanism")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(os.environ.get("IBCI_DATA", "D:/ibci-data"), "braingate")
    yroot = os.environ.get("IBCI_YIELD", "D:/ibci-data/braingate")   # folder that contains yield/<P>/
    lin = pd.read_csv(f"results/replication_bg/{args.subject}_ladder.csv")
    lin = lin[lin.gap_target.isin(GAPS)][["train", "test", "gap_target", "days"]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    trains = sorted(lin.train.unique())
    keep = set(rng.choice(trains, min(args.max_train, len(trains)), replace=False))
    pairs = lin[lin.train.isin(keep)].reset_index(drop=True)
    sess = sessions_for(args.subject, 1e4, set(pairs.train) | set(pairs.test))
    rows = []
    for _, r in pairs.iterrows():
        si, sj = sess[r.train], sess[r.test]
        yi, yj = yield_day(yroot, args.subject, si.day), yield_day(yroot, args.subject, sj.day)
        if yi is None or yj is None:
            continue
        g = fit_gains(si.dec, sj)
        n_ch = len(g)
        ids = np.arange(n_ch)
        d_rate = np.log(yj.rate.reindex(ids).to_numpy(float) + 0.1) - np.log(yi.rate.reindex(ids).to_numpy(float) + 0.1)
        d_imp = np.log(yj.impedance.reindex(ids).to_numpy(float)) - np.log(yi.impedance.reindex(ids).to_numpy(float))
        row = dict(subject=args.subject, **r.to_dict())
        for name, x in (("rate", d_rate), ("imp", d_imp)):
            ok = np.isfinite(x)
            if ok.sum() >= 20:
                row[f"rho_{name}"] = float(stats.spearmanr(g[ok], x[ok]).correlation)
                row[f"rho_{name}_null"] = float(stats.spearmanr(g[ok], rng.permutation(x[ok])).correlation)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.subject}.csv"), index=False)
    for name in ("rate", "imp"):
        c = f"rho_{name}"
        if c in df and df[c].notna().sum() >= 6:
            s = df.groupby("train")[c].mean().dropna()
            w = stats.wilcoxon(s)
            print(f"{args.subject} {name}: pairs {df[c].notna().sum()}, sessions {len(s)}, median rho {s.median():.3f}, "
                  f"positive {np.mean(s > 0):.2f}, P {w.pvalue:.2g}; null median {df[c + '_null'].median():.3f}")


if __name__ == "__main__":
    main()
