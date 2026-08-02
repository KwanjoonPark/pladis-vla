#!/bin/bash
# One-screen view of a sweep campaign, replacing `tail` over 38 scattered .out files.
#
#   bash scripts/sweep_status.sh                 # arm x suite matrix + live arms + GPUs
#   bash scripts/sweep_status.sh --logs          # + the last progress line of every arm
#   bash scripts/sweep_status.sh --watch         # refresh every 60s
#   bash scripts/sweep_status.sh --prefix pi05_orig
#
# Truth comes from the EPLOGS, not the logs: an eplog row is a flushed episode, whereas a
# .out file is overwritten on every driver restart (so a resumed arm's log says
# "DONE 0 eps" and carries no history). Progress is therefore counted, not parsed.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREFIX=pi05_lang
SHOW_LOGS=0
WATCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --logs)   SHOW_LOGS=1; shift ;;
    --watch)  WATCH=1; shift ;;
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown flag '$1' (see --help)" >&2; exit 2 ;;
  esac
done

SUITES="libero_10 libero_goal libero_object libero_spatial"

# Curated episode count per suite = the vanilla arm's row count. Read rather than
# hardcoded, so this stays correct if the curated list or the axis changes.
declare -A TOTAL
for s in $SUITES; do
  f="results/sweep/${PREFIX}_vanilla_${s}_eplog.tsv"
  TOTAL[$s]=$([ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0)
done

# Arm order: the driver's own `run` lines, so the display matches execution order and
# picks up new arms automatically instead of needing a second list to maintain.
arms_of_prefix() {
  local drv
  case "$PREFIX" in
    pi05_lang) drv=experiments/sweep_pi05_language.sh ;;
    pi05_orig) drv=experiments/sweep_pi05_original.sh ;;
    *)         drv="" ;;
  esac
  if [ -n "$drv" ] && [ -f "$drv" ]; then
    grep -oP '^run\s+\K[A-Za-z0-9_]+' "$drv"
  else
    # fall back to whatever eplogs exist
    ls results/sweep/${PREFIX}_*_eplog.tsv 2>/dev/null \
      | sed -E "s|.*/${PREFIX}_(.+)_(libero_[a-z0-9]+)_eplog\.tsv|\1|" | sort -u
  fi
}

render() {
  printf '\n\033[1m%s\033[0m   %s\n' "sweep status: $PREFIX" "$(date '+%m-%d %H:%M:%S')"

  # ---- running arms (which suite is on which device, and how far along)
  printf '\n\033[1mrunning\033[0m\n'
  local any=0
  while read -r pid dev args; do
    [ -z "${pid:-}" ] && continue
    local suite arm
    suite=$(grep -oP '(?<=--suite )\S+' <<<"$args")
    arm=$(grep -oP "(?<=/${PREFIX}_)[A-Za-z0-9_]+(?=_${suite}_eplog)" <<<"$args" | head -1)
    local out="results/sweep/${PREFIX}_${arm}_${suite}.out"
    local last
    last=$(grep -h 'running-SR\|DONE' "$out" 2>/dev/null | tail -1 | sed 's/^\[arm\] //')
    printf '  GPU %-3s %-15s %-12s %s\n' "$dev" "$suite" "$arm" "${last:-starting up}"
    any=1
  done < <(
    for p in $(pgrep -f "eval_arm.py --model pi05" 2>/dev/null); do
      local_args=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
      dev=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -oP '(?<=^CUDA_VISIBLE_DEVICES=).*')
      echo "$p ${dev:-?} $local_args"
    done
  )
  [ "$any" = 0 ] && echo "  (no arm running)"

  # ---- arm x suite matrix: done/total and SR
  printf '\n\033[1mprogress\033[0m   n/total  (SR%%)\n'
  printf '  %-14s' 'arm'
  for s in $SUITES; do printf '%18s' "${s#libero_}"; done
  printf '\n'
  for a in $(arms_of_prefix); do
    printf '  %-14s' "$a"
    for s in $SUITES; do
      local f="results/sweep/${PREFIX}_${a}_${s}_eplog.tsv"
      if [ ! -f "$f" ]; then
        printf '%18s' '-'
      else
        awk -F'\t' -v tot="${TOTAL[$s]}" '
          NR>1 { n++; ok += $6 }
          END {
            sr = n ? 100*ok/n : 0
            mark = (tot && n >= tot) ? "" : "*"
            printf "%18s", sprintf("%d/%d (%.1f)%s", n, tot, sr, mark)
          }' "$f"
      fi
    done
    printf '\n'
  done
  echo "  ( * = incomplete )"

  # ---- drivers and devices
  printf '\n\033[1mdrivers\033[0m\n'
  for s in $SUITES; do
    local d="results/sweep/driver_${PREFIX#pi05_}_${s}.out"
    [ -f "$d" ] || continue
    printf '  %-15s %s\n' "$s" "$(tail -1 "$d")"
  done

  printf '\n\033[1mGPUs\033[0m  (mine = holding a pi05 eval_arm)\n'
  local mine
  mine=$(for p in $(pgrep -f "eval_arm.py --model pi05" 2>/dev/null); do
           tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
             | grep -oP '(?<=^CUDA_VISIBLE_DEVICES=).*'
         done | sort -u | paste -sd, -)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while IFS=, read -r i m; do
    i=$(tr -d ' ' <<<"$i")
    local tag=''
    [[ ",${mine}," == *",${i},"* ]] && tag='  <- mine'
    printf '  GPU %-3s %10s%s\n' "$i" "$(tr -d ' ' <<<"$m")" "$tag"
  done

  if [ "$SHOW_LOGS" = 1 ]; then
    printf '\n\033[1mlast line per arm\033[0m\n'
    for a in $(arms_of_prefix); do
      for s in $SUITES; do
        local out="results/sweep/${PREFIX}_${a}_${s}.out"
        [ -f "$out" ] || continue
        printf '  %-14s %-15s %s\n' "$a" "${s#libero_}" \
          "$(grep -h 'running-SR\|DONE\|Traceback\|Error' "$out" 2>/dev/null | tail -1 | sed 's/^\[arm\] //')"
      done
    done
  fi
  echo
}

if [ "$WATCH" = 1 ]; then
  while true; do clear; render; sleep 60; done
else
  render
fi
