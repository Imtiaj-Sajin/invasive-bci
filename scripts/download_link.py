"""Download the LINK dataset (DANDI 001201, CC-BY-4.0, ~12.6 GB, 312 NWB sessions).

Temmar et al., "LINK: Long-term Intracortical Neural activity and Kinematics", NeurIPS D&B 2025.
Files are saved as <root>/link/<session>.nwb. Already-complete files are skipped, so the script can be re-run.

Usage:
    python scripts/download_link.py [--root D:/ibci-data] [--workers 4]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ASSETS_URL = "https://api.dandiarchive.org/api/dandisets/001201/versions/draft/assets/?page_size=400"
DOWNLOAD_URL = "https://api.dandiarchive.org/api/assets/{}/download/"


def fetch(asset, out_dir):
    name = os.path.basename(asset["path"])
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) == asset["size"]:
        return name, "skip"
    tmp = dst + ".part"
    with requests.get(DOWNLOAD_URL.format(asset["asset_id"]), stream=True, timeout=120) as r:
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
    ap.add_argument("--root", default=os.environ.get("IBCI_DATA", "D:/ibci-data"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.join(args.root, "link")
    os.makedirs(out_dir, exist_ok=True)
    assets = requests.get(ASSETS_URL, timeout=60).json()["results"]
    print(f"{len(assets)} assets, {sum(a['size'] for a in assets) / 1e9:.2f} GB -> {out_dir}", flush=True)

    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, a, out_dir) for a in assets]
        for fut in as_completed(futs):
            try:
                name, status = fut.result()
            except Exception as e:  # keep going; re-run the script to retry failures
                print("FAILED", e, flush=True)
                continue
            done += 1
            if done % 20 == 0 or done == len(assets):
                print(f"{done}/{len(assets)} {name} {status}", flush=True)


if __name__ == "__main__":
    main()
