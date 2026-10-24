#!/usr/bin/env bash
# Run the baseline on the college GPU inside a time-limited slot.
#
#   ./run_server.sh --until 00:00     full 3B baseline (add --with-1b for the 1B comparison afterwards;
#                                     needs WITH_1B=1 ./setup_server.sh once to download the 1B model)
#                                     HH:MM = when the slot ENDS; new work stops 12 minutes before that.
#   ./run_server.sh --minutes 80      same, with a budget in minutes instead of an end time
#   ./run_server.sh --quick           3-5 minute smoke test on the real 3B model
#   ./run_server.sh --until 00:00 --no-1b      skip the 1B comparison
#   ./run_server.sh --until 00:00 --only-1b    only the 1B comparison
#
# Results are written after every single configuration, so if the slot ends (or the SSH connection drops) nothing
# that finished is lost, and the next run resumes where this one stopped. Keep the terminal open: logging out stops
# the run (nohup/screen/tmux do not help on this server).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
HERE="$PWD"
if [ ! -x .venv/bin/python ]; then echo "Run ./setup_server.sh first."; exit 1; fi
VPY="$HERE/.venv/bin/python"
export HF_HOME="$HERE/.hf_cache" PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
if [ -f .prefetch_ok ] && [ -z "${KV_ONLINE:-}" ]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi

UNTIL="" MINUTES="" QUICK="" RUN3B=1 RUN1B=0 RESERVE_MIN=12
CONFIG_3B="${KV_CONFIG_3B:-configs/server_3b.yaml}"   # overridable for testing
CONFIG_1B="${KV_CONFIG_1B:-configs/server_1b.yaml}"
while [ $# -gt 0 ]; do
  case "$1" in
    --until) UNTIL="$2"; shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --quick) QUICK="--quick"; shift ;;
    --no-1b) RUN1B=0; shift ;;
    --with-1b) RUN1B=1; shift ;;
    --only-1b) RUN3B=0; RUN1B=1; shift ;;
    --reserve) RESERVE_MIN="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
if [ -z "$UNTIL" ] && [ -z "$MINUTES" ] && [ -z "$QUICK" ]; then
  echo "[warn] no --until/--minutes given: using an 85-minute budget. Better: ./run_server.sh --until <slot end HH:MM>"
  MINUTES=85
fi

# absolute deadline (unix seconds) for starting new work
DEADLINE=$("$VPY" - "$UNTIL" "$MINUTES" "$RESERVE_MIN" <<'PY'
import sys, time, datetime as dt
until, minutes, reserve = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = []
if until:
    now = dt.datetime.now()
    h, m = map(int, until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    c.append(end.timestamp() - 60 * reserve)
if minutes:
    c.append(time.time() + 60 * float(minutes))
print(int(min(c)) if c else 0)
PY
)
if [ -n "$QUICK" ] && [ "$DEADLINE" = "0" ]; then DEADLINE=$(( $(date +%s) + 20 * 60 )); fi
LEFT=$(( (DEADLINE - $(date +%s)) / 60 ))
if [ "$LEFT" -le 3 ]; then echo "[FAIL] only $LEFT min left before the deadline; nothing started."; exit 1; fi

# pick the GPU with the most free memory if several are visible
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$(nvidia-smi -L | wc -l)" -gt 1 ]; then
  GPU_IDX=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  export CUDA_VISIBLE_DEVICES="$GPU_IDX"
  echo "[gpu] several GPUs visible; using GPU $CUDA_VISIBLE_DEVICES (most free memory)"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/run_${STAMP}.log"
echo "[run] new work stops at $(date -d @"$DEADLINE" +%H:%M:%S) ($LEFT min from now). Log: $LOG"

if [ "$RUN3B" = 1 ]; then
  "$VPY" scripts/run_all.py --config "$CONFIG_3B" --deadline "$DEADLINE" $QUICK 2>&1 | tee -a "$LOG"
fi
LEFT=$(( (DEADLINE - $(date +%s)) / 60 ))
if [ "$RUN1B" = 1 ] && [ -z "$QUICK" ]; then
  if [ "$LEFT" -ge 10 ]; then
    echo -e "\n######## 1B comparison run ($LEFT min left) ########" | tee -a "$LOG"
    "$VPY" scripts/run_all.py --config "$CONFIG_1B" --deadline "$DEADLINE" 2>&1 | tee -a "$LOG"
  else
    echo "[skip] 1B comparison: only $LEFT min left" | tee -a "$LOG"
  fi
fi
"$VPY" scripts/compare_runs.py 2>&1 | tee -a "$LOG"

ARCH="results_${STAMP}.tar.gz"
tar czf "$ARCH" results logs 2>/dev/null
echo
echo "Packed results: $HERE/$ARCH"
echo "Copy to your laptop (run this ON YOUR LAPTOP while the slot is still active):"
echo "  scp $(whoami)@10.4.25.63:$HERE/$ARCH ."
