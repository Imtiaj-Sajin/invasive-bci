# Running one BrainGate participant on a second PC (T11)

T11 is the largest participant (21.2 GB archive, ~290 yield sessions). Its analyses need more memory than the main PC
(24 GB) can give while other jobs run, so it can run on a second machine. Results use the same file names and can be
committed and merged as usual.

## 1. Get the code and Python packages

```bash
git clone https://github.com/Imtiaj-Sajin/invasive-bci.git
cd invasive-bci
python -m venv .venv
.venv/Scripts/python -m pip install -e .            # numpy, scipy, pandas, scikit-learn, matplotlib
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA 12.8 builds
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On Linux or macOS use `.venv/bin/python` instead of `.venv/Scripts/python`.

## 2. Data

- **Decoding data.** Either pass the server URL (streamed, nothing stored), or the downloaded
  `decoding_T11.tar.gz`. Conversion writes compact files to `<DATA_ROOT>/braingate/decoding/T11/` (~18–20 GB).
- **Yield data** (only for the gain–electrode link): copy the folder `D:/ibci-data/braingate/yield/T11` (936 MB,
  292 files) from the main PC, or download `yield_T11.tar.gz` from Dryad (doi:10.5061/dryad.x0k6djj1h) and extract it.
  Put it at `<YIELD_ROOT>/yield/T11/`.

## 3. Run

```bash
# with the archive already downloaded:
bash scripts/run_participant_standalone.sh T11 D:/ibci-data D:/downloads/decoding_T11.tar.gz D:/ibci-data/braingate
# or streaming from the server:
bash scripts/run_participant_standalone.sh T11 D:/ibci-data http://31.97.211.4:53808/decoding_T11.tar.gz D:/ibci-data/braingate
```

The script converts the archive (checks the gzip CRC), then runs the correction ladder, the recurrent-network ladder
on the GPU, and the other analyses one at a time. Each finished step writes a `.ok` marker, so it can be restarted
safely. Logs are in `results/new_participants/T11_*.log`.

## 4. Return the results

```bash
git add results/new_participants results/replication_bg/T11_ladder.csv results/*/T11*.csv results/reg_tradeoff_T11
git -c user.name="Imtiaj Sajin" -c user.email="imtiajsajin@gmail.com" commit -m "Add results for BrainGate participant T11"
git push
```

Then tell the main session, which will stop its own T11 run and merge the results.
