"""Resumable multi-connection HTTP download using byte ranges.

Splits the file into fixed-size chunks, downloads them with N parallel connections straight into a preallocated
file, and records finished chunks in <dest>.progress so an interrupted run resumes where it stopped.

Usage: python scripts/tools/fetch_parallel.py URL DEST [--workers 8] [--chunk-mb 64] [--expect-bytes N]
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def total_size(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    r.close()
    return int(r.headers["Content-Range"].split("/")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--expect-bytes", type=int, default=None)
    args = ap.parse_args()

    size = total_size(args.url)
    if args.expect_bytes and size != args.expect_bytes:
        raise SystemExit(f"server reports {size} bytes, expected {args.expect_bytes}")
    chunk = args.chunk_mb << 20
    chunks = [(i, s, min(s + chunk, size) - 1) for i, s in enumerate(range(0, size, chunk))]
    prog_path = args.dest + ".progress"
    done = set(json.load(open(prog_path))) if os.path.exists(prog_path) else set()
    if not os.path.exists(args.dest):
        with open(args.dest, "wb") as f:
            f.truncate(size)
    lock = threading.Lock()
    t0, got = time.time(), [0]

    def fetch(i, a, b):
        for attempt in range(8):
            try:
                with requests.get(args.url, headers={"Range": f"bytes={a}-{b}"}, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    buf = r.content
                if len(buf) != b - a + 1:
                    raise IOError(f"chunk {i}: got {len(buf)} bytes")
                with lock:
                    with open(args.dest, "r+b") as f:
                        f.seek(a)
                        f.write(buf)
                    done.add(i)
                    got[0] += len(buf)
                    json.dump(sorted(done), open(prog_path, "w"))
                return i
            except Exception as e:  # retry with backoff
                time.sleep(2 * (attempt + 1))
                err = e
        raise err

    todo = [c for c in chunks if c[0] not in done]
    print(f"{size} bytes, {len(chunks)} chunks, {len(todo)} to fetch, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, *c) for c in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if k % 5 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"{len(done)}/{len(chunks)} chunks, {got[0] / el / 1e6:.1f} MB/s, {el:.0f}s", flush=True)
    if len(done) == len(chunks) and os.path.getsize(args.dest) == size:
        os.remove(prog_path)
        print("complete:", args.dest, size, "bytes", flush=True)


if __name__ == "__main__":
    main()
