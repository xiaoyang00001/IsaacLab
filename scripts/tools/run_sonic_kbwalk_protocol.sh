#!/bin/bash
# 键盘行走协议 3v3：τ0 vs τ10，脚本化按键序列保证每轮动作一致（headless）
set -u
WT=/home/nolo/xiaoyang_IssacLab/IsaacLab-sonic-v3-eval-opt-20260716
cd "$WT"

run_round() {
  local tau="$1" label="$2"
  while pgrep -f "run_sonic_jitter_closed_loop.sh" >/dev/null 2>&1; do sleep 5; done
  sleep 10
  echo "===== ROUND $label (tau=$tau) $(date +%H:%M:%S) ====="
  SONIC_ENV_PHASE_LOCK=1 SONIC_STATE_OBS_LEAD_S=$tau JITTER_INPUT=keyboard JITTER_GUI=0 \
    scripts/tools/run_sonic_jitter_closed_loop.sh "$label" -- --free_seconds 150 &
  local orch=$!
  local rundir=""
  for i in $(seq 1 90); do
    rundir=$(ls -dt /tmp/sonic_jitter/${label}_* 2>/dev/null | head -1)
    [ -n "$rundir" ] && [ -f "$rundir/isaac.log" ] && break
    sleep 2
  done
  [ -z "$rundir" ] && { echo "$label RUNDIR_NOT_FOUND"; wait $orch; return 1; }
  for i in $(seq 1 240); do
    grep -q "start locked recording" "$rundir/isaac.log" 2>/dev/null && break
    sleep 2
  done
  echo "$label locked detected; +45s 开始按键"
  sleep 45
  local ses=$(tmux ls 2>/dev/null | grep -o "sonic_jitter_[0-9]*" | head -1)
  [ -z "$ses" ] && { echo "$label TMUX_NOT_FOUND"; wait $orch; return 1; }
  DK() { tmux send-keys -t "$ses:deploy" "$1"; }
  DK '1'; sleep 2
  DK '0'; sleep 1; DK '0'; sleep 1; DK '0'; sleep 1; DK '0'; sleep 2
  for i in $(seq 1 15); do DK 'w'; sleep 1; done
  DK 'r'; sleep 6
  DK '1'; sleep 1
  for i in $(seq 1 15); do DK 's'; sleep 1; done
  DK 'r'
  echo "$label 按键序列完毕 $(date +%H:%M:%S)"
  wait $orch
  echo "$label DONE $(date +%H:%M:%S)"
}

run_round 0     kwq_a1
run_round 0.010 kwq_b1
run_round 0.010 kwq_b2
run_round 0     kwq_a2
run_round 0     kwq_a3
run_round 0.010 kwq_b3
echo "KWQ_3V3_DONE $(date +%H:%M:%S)"
