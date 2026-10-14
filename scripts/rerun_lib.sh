# Shared helpers for the audit rerun scripts (sourced, not run). Two rules keep parallel lanes from running out of
# memory (out-of-memory failures on 2026-10-13 and 2026-10-14):
#   1. Monkeys C and M load all their sessions at once (about 9 GB each while loading), so only one C or M step runs at
#      a time across all lanes (global lock directory $OUT/heavy.lock).
#   2. A step starts only when enough commit memory is free (RAM + page file headroom, the limit that was hit).

commit_free_gb() { powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" | tr -dc 0-9; }

need_gb() { case "$1" in C|M) echo 12 ;; *) echo 6 ;; esac; }

heavy_acquire() {   # heavy_acquire SUBJECT
  case "$1" in C|M) until mkdir "$OUT/heavy.lock" 2>/dev/null; do sleep 60; done ;; esac
}

heavy_release() {
  case "$1" in C|M) rmdir "$OUT/heavy.lock" 2>/dev/null ;; esac
}

wait_mem() {        # wait_mem SUBJECT
  while [ "$(commit_free_gb)" -lt "$(need_gb "$1")" ]; do sleep 60; done
}
