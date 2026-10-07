"""Stream a BrainGate decoding archive (decoding_<P>.tar.gz) into compact per-session .mat files.

The full archives are 6-12 GB each. Each session file holds spike-band power plus threshold crossings at six
thresholds in float64. Our analyses need only spike-band power, cursor and target positions, trial and block indices
and electrode metadata. This tool reads the archive as a stream (no full extraction), keeps those fields plus the
-4.5 RMS threshold crossings, casts neural data to float32 and writes compressed .mat files with the same structure,
so scripts/replicate_braingate_decoding.py reads them unchanged. Reading the stream to the end also checks the gzip CRC.

ARCHIVE may be a local path or an http(s) URL; a URL is streamed straight into the converter, so the archive is never
stored. If the connection drops, the stream restarts from the beginning and sessions already written are skipped.

Usage: python scripts/tools/compact_braingate_decoding.py ARCHIVE OUT_DIR
"""
import argparse
import io
import os
import tarfile
import time

import numpy as np
from scipy.io import loadmat, savemat

KEEP_NEURAL = ("sbp", "tx_4_5")


def compact(raw):
    d = loadmat(io.BytesIO(raw), simplify_cells=True)
    out = {k: d[k] for k in ("participant_id", "post_implant_day", "bin_size", "electrodes", "blocks", "trials")
           if k in d}
    out["cursor_position"] = np.asarray(d["cursor_position"], dtype=np.float32)
    out["target_position"] = np.asarray(d["target_position"], dtype=np.float32)
    out["neural"] = {k: np.asarray(d["neural"][k], dtype=np.float32) for k in KEEP_NEURAL if k in d["neural"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for attempt in range(1, 6):
        try:
            convert(args.archive, args.out_dir)
            return
        except (OSError, EOFError, tarfile.TarError) as e:   # dropped connection or truncated stream: start again,
            print(f"attempt {attempt} failed: {e!r}", flush=True)  # sessions already written are skipped
            if not args.archive.startswith("http") or attempt == 5:
                raise
            time.sleep(30)


def convert(archive, out_dir):
    """Read the archive (a local path or an http(s) URL, streamed without saving the archive) to the end."""
    n, t0, read_bytes = 0, time.time(), 0
    if archive.startswith("http"):
        import urllib.request
        src = urllib.request.urlopen(archive, timeout=120)
        tar_ctx = tarfile.open(fileobj=src, mode="r|gz")
    else:
        tar_ctx = tarfile.open(archive, "r|gz")
    with tar_ctx as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".mat")):
                continue
            dest = os.path.join(out_dir, os.path.basename(m.name))
            raw = tar.extractfile(m).read()
            read_bytes += len(raw)
            if not os.path.exists(dest):
                savemat(dest + ".part", compact(raw), do_compression=True, appendmat=False)
                os.replace(dest + ".part", dest)
            n += 1
            print(f"{n:4d} {os.path.basename(m.name)}  {read_bytes / 1e9:.1f} GB read, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} sessions; archive read to the end (gzip CRC verified)", flush=True)


if __name__ == "__main__":
    main()
