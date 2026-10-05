"""Download NWB assets of a DANDI dandiset (optionally only some subjects) into <root>/<dest>/.

Already-complete files are skipped, so the script can be re-run after an interruption.

Usage:
    python scripts/download_dandi.py --dandiset 000688 --subjects C M --dest perich [--root D:/ibci-data] [--workers 4]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://api.dandiarchive.org/api"


def list_assets(dandiset):
    url = f"{API}/dandisets/{dandiset}/versions/draft/assets/?page_size=400"
    out = []
    while url:
        r = requests.get(url, timeout=60).json()
        out += r["results"]
        url = r.get("next")
    return out


def fetch(asset, out_dir):
    name = os.path.basename(asset["path"])
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) == asset["size"]:
        return name, "skip"
    tmp = dst + ".part"
    with requests.get(f"{API}/assets/{asset['asset_id']}/download/", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if os.path.getsize(tmp) != asset["size"]:
        raise IOError(f"size mismatch for {name}")
    os.replace(tmp, dst)
    return name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dandiset", required=True)
    ap.add_argument("--subjects", nargs="*", default=None, help="keep assets under sub-<S>/ for these S")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--root", default=os.environ.get("IBCI_DATA", "D:/ibci-data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.join(args.root, args.dest)
    os.makedirs(out_dir, exist_ok=True)
    assets = [a for a in list_assets(args.dandiset)
              if args.subjects is None or any(a["path"].startswith(f"sub-{s}/") for s in args.subjects)]
    print(f"{len(assets)} assets, {sum(a['size'] for a in assets) / 1e9:.2f} GB -> {out_dir}", flush=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, a, out_dir) for a in assets]):
            try:
                name, status = fut.result()
            except Exception as e:  # re-run to retry
                print("FAILED", e, flush=True)
                continue
            done += 1
            if done % 10 == 0 or done == len(assets):
                print(f"{done}/{len(assets)} {name} {status}", flush=True)


if __name__ == "__main__":
    main()
