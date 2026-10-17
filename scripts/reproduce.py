"""Reproduce the study end to end, stage by stage.

Every analysis in the paper is one command below, in the order in which results depend on each other. Each command
writes its own folder under results/; a stage can be rerun on its own. Logs go to logs/<stage>/.

    python scripts/reproduce.py --list                # show the stages and their commands
    python scripts/reproduce.py data                  # download the DANDI datasets; print the manual Dryad steps
    python scripts/reproduce.py ladders               # correction ladders (default ridge penalty)
    python scripts/reproduce.py analyses              # tuned ladders, alignment ceiling, controls, recalibration
    python scripts/reproduce.py electrodes            # electrode statistics, MINDFUL re-analysis, monitoring
    python scripts/reproduce.py simulator             # simulator calibration, validation and augmentation
    python scripts/reproduce.py summaries             # summaries, statistics, benchmark baselines and figures
    python scripts/reproduce.py check                 # unit tests and benchmark reproduction check
    python scripts/reproduce.py all                   # everything above, in order

Options: --dry-run prints the commands without running them; --skip-existing skips a command whose output folder
already exists; --subjects limits the per-individual commands (e.g. --subjects T6 T5).
Where the data are stored is set in one place: see src/ibci/paths.py and docs/REPRODUCE.md.
The NDT3 foundation-decoder pilot needs a separate Linux/WSL environment and is described in docs/REPRODUCE.md.
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MONKEYS = ["N", "C", "M"]
HUMANS = ["T6", "T5", "T9", "T7", "T10", "T8"]          # included by the pre-specified rule
EXCLUDED = ["T2", "T3", "T11"]                         # analyzed for the Supplementary Information only
ALL9 = MONKEYS + HUMANS


def ladders(sub):
    cmds = []
    if "N" in sub:
        cmds.append((["scripts/drift_anatomy.py", "--max-train", "80", "--out", "results/anatomy"], "results/anatomy"))
    if {"C", "M"} & set(sub):
        cmds.append((["scripts/replicate_perich.py", "--subjects", "C", "M", "--out", "results/replication"],
                     "results/replication"))
    for p in [p for p in HUMANS + EXCLUDED if p in sub]:
        cmds.append((["scripts/replicate_braingate_decoding.py", "--participant", p, "--skip", "L4", "L5",
                      "--resume"], f"results/replication_bg/{p}_ladder.csv"))
    if "T11" in sub:   # sensitivity analysis: each recording block z-scored separately
        cmds.append((["scripts/replicate_braingate_decoding.py", "--participant", "T11", "--skip", "L4", "L5",
                      "--out", "results/replication_bg_blocknorm", "--env", "IBCI_BLOCKNORM=1"],
                     "results/replication_bg_blocknorm/T11_ladder.csv"))
    return cmds


def analyses(sub):
    cmds = []
    for k in [k for k in ALL9 if k in sub]:
        for s in ("decay_alpha", "latent_ceiling", "reweight_controls", "align_variants", "label_free_plus",
                  "weight_angles", "nn_ladder"):
            out = {"decay_alpha": f"results/decay_alpha/{k}_alpha10000.csv"}.get(s, f"results/{s}/{k}.csv")
            cmds.append(([f"scripts/{s}.py", "--subject", k], out))
        if k == "N":
            cmds.append((["scripts/reg_tradeoff.py", "--dataset", "link", "--out", "results/reg_tradeoff"],
                         "results/reg_tradeoff"))
        elif k in ("C", "M"):
            cmds.append((["scripts/reg_tradeoff.py", "--dataset", "perich", "--subject", k, "--out",
                          f"results/reg_tradeoff_{k}"], f"results/reg_tradeoff_{k}"))
            cmds.append((["scripts/matched_output_control.py", "--subject", k, "--alpha", "1e4", "--out",
                          "results/matched_output_tuned"], f"results/matched_output_tuned/{k}.csv"))
        else:
            cmds.append((["scripts/reg_tradeoff.py", "--dataset", "braingate", "--subject", k, "--out",
                          f"results/reg_tradeoff_{k}"], f"results/reg_tradeoff_{k}"))
            cmds.append((["scripts/gain_mechanism.py", "--subject", k], f"results/gain_mechanism/{k}.csv"))
            cmds.append((["scripts/gain_efficiency.py", "--subject", k], f"results/gain_efficiency/{k}.csv"))
    if "N" in sub:
        cmds += [(["scripts/drift_anatomy.py", "--gaps", "1", "7", "30", "120", "480", "--n-trials", "10", "20", "50",
                   "100", "300", "--max-train", "30", "--out", "results/data_efficiency"], "results/data_efficiency"),
                 (["scripts/recal_policy.py"], "results/recal_policy")]
    return cmds


def electrodes(sub):
    return [(["scripts/channel_health.py", "--out", "results/channel_health"], "results/channel_health"),
            (["scripts/braingate_failure.py"], "results/braingate_failure"),
            (["scripts/failure_robustness.py"], "results/failure_robustness/summary.json"),
            (["scripts/failure_null.py"], "results/failure_robustness/null_summary.json"),
            (["scripts/failure_stats_xdata.py", "--datasets", "link", "perich"], "results/failure_xdata"),
            (["scripts/amplitude_summary.py"], "results/amplitude_T5.json"),
            (["scripts/decoding_session_stats.py"], "results/decoding_session_stats.json"),
            (["scripts/mindful_reanalysis.py"], "results/mindful_reanalysis"),
            (["scripts/mindful_test.py"], "results/mindful_test")]


def simulator(sub):
    return [(["scripts/failure_stats_xdata.py", "--datasets", "link", "--max-day", "700", "--out",
              "results/failure_xdata_cal700"], "results/failure_xdata_cal700"),
            (["scripts/failure_stats_xdata.py", "--datasets", "link", "--min-day", "700", "--out",
              "results/failure_xdata_late700"], "results/failure_xdata_late700"),
            (["scripts/calibrate_sim.py", "--max-day", "700", "--failure-json",
              "results/failure_xdata_cal700/failure_stats.json", "--out", "results/sim_calib_split"],
             "results/sim_calib_split"),
            (["scripts/calibrate_sim.py", "--min-day", "700", "--failure-json",
              "results/failure_xdata_late700/failure_stats.json", "--out", "results/sim_calib_late"],
             "results/sim_calib_late"),
            (["scripts/validate_sim.py"], "results/sim_validation"),
            (["scripts/sim_augment.py", "--calib", "results/sim_calib_split/calibration.json", "--split-day", "700",
              "--out", "results/augment"], "results/augment")]


def summaries(sub):
    cmds = [(["scripts/v2_summary.py"], None),
            (["scripts/v2_summary.py", "--env", "IBCI_ALL_PAIRS=1"], None),
            (["scripts/decay_summary.py"], None),
            (["scripts/decay_summary.py", "--tuned"], None),
            (["scripts/heterogeneity_model.py"], None),
            (["scripts/rule_summary.py"], None),
            (["scripts/paper_stats.py"], None),
            (["scripts/bench_from_paper.py"], None),
            (["scripts/ndt3/summarize_pilot.py"], None),
            (["scripts/revision_stats.py"], None),
            (["scripts/ceiling_summary.py"], None),
            (["scripts/make_v2_figures.py", "--only", "fig2"], None),    # session overview read by the next step
            (["scripts/make_numbers_tex.py", "--out", "results/numbers.tex"], None),   # every number quoted in the paper
            (["scripts/make_supp_tables.py", "--out", "results/supp_tables"], None),   # supplementary tables
            (["scripts/make_v2_figures.py"], None)]
    if os.path.isdir(os.path.join(REPO, "manuscript", "natcomms")):     # local manuscript sources (not public yet)
        cmds += [(["scripts/make_numbers_tex.py"], None), (["scripts/make_supp_tables.py"], None)]
    return cmds


def check(sub):
    return [(["-m", "pytest", "tests", "-q"], None),
            (["scripts/bench_check.py", "--subjects", "C", "M", "T6", "--pairs", "2"], None)]


STAGES = {"ladders": ladders, "analyses": analyses, "electrodes": electrodes, "simulator": simulator,
          "summaries": summaries, "check": check}

DATA_HELP = """Manual downloads (Dryad requires a browser; files are free, licence CC0):
  1. BrainGate decoding and yield data, Hahn et al. 2026: https://doi.org/10.5061/dryad.x0k6djj1h
     Put decoding_<P>.tar.gz and the yield archive in DATA/downloads/. Convert each decoding archive with
        python scripts/tools/compact_braingate_decoding.py DATA/downloads/decoding_<P>.tar.gz DATA/braingate/decoding/<P>
     and extract the yield files to DATA/braingate/yield/<P>/.
  2. MINDFUL closed-loop data, Pun et al. 2024: https://doi.org/10.5061/dryad.n2z34tn5s
     Extract to DATA/mindful/MINDFUL_Data/.
DATA is the data folder chosen in src/ibci/paths.py (default: data/ in this repository)."""


def run(stage, cmd, out, args):
    env, cmd, prefix = dict(os.environ), list(cmd), ""
    if "--env" in cmd:                                  # internal marker: environment variable for this command
        i = cmd.index("--env")
        key, val = cmd[i + 1].split("=", 1)
        env[key], prefix = val, f"{key}={val} "
        del cmd[i:i + 2]
    full = [PY] + cmd
    shown = prefix + " ".join(cmd)
    if args.skip_existing and out and os.path.exists(os.path.join(REPO, out)):
        print(f"  skip  {shown}   ({out} exists)")
        return True
    if args.dry_run:
        print(f"  {shown}")
        return True
    os.makedirs(os.path.join(REPO, "logs", stage), exist_ok=True)
    name = "_".join([prefix.strip().replace("=", "-")] * bool(prefix) + [x.lstrip("-") for x in cmd])
    name = name.replace("scripts/", "").replace("results/", "").replace("/", "_").replace(".py", "")
    log = os.path.join(REPO, "logs", stage, f"{name[:120]}.log")
    t0 = time.time()
    print(f"  run   {shown}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.call(full, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"        {'done' if code == 0 else 'FAILED'} in {time.time() - t0:.0f} s (log: {os.path.relpath(log, REPO)})",
          flush=True)
    return code == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default=None, choices=["data", *STAGES, "all"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--subjects", nargs="+", default=ALL9 + EXCLUDED)
    args = ap.parse_args()
    if args.list or args.stage is None:
        args.dry_run = True
        stages = list(STAGES)
    elif args.stage == "all":
        stages = ["data", *STAGES]
    else:
        stages = [args.stage]
    failed = []
    for st in stages:
        print(f"== {st}")
        if st == "data":
            for cmd in (["scripts/download_link.py", "--workers", "8"],
                        ["scripts/download_dandi.py", "--dandiset", "000688", "--subjects", "C", "M", "--dest",
                         "perich"]):
                if not run("data", cmd, None, args):
                    failed.append(" ".join(cmd))
            print(DATA_HELP)
            continue
        for cmd, out in STAGES[st](args.subjects):
            if not run(st, cmd, out, args):
                failed.append(" ".join(cmd))
    if failed:
        print("\nFailed commands (see logs/):\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
