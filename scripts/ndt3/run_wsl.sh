#!/usr/bin/env bash
# Run a Python script in the NDT3 WSL environment with a log file and an exit marker.
# Usage (from Windows Git Bash): MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -- bash "/mnt/g/codes/invasive bci/scripts/ndt3/run_wsl.sh" LOG SCRIPT [ARGS...]
log=$1; shift
cd /opt/ndt3 || exit 1
/opt/ndt3-venv/bin/python -u "$@" > "$log" 2>&1
echo "EXIT=$?" >> "$log"
