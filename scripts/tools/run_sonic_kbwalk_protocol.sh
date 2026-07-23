#!/bin/bash
# 键盘行走协议（协议三）：tmux 自动按键驱动，可单轮可全量 3v3，三种步态。
#
# 用法：
#   run_sonic_kbwalk_protocol.sh                # 全量 3v3：lock+τ0 ×3 vs lock+τ10 ×3（默认无头、slow 步态）
#   run_sonic_kbwalk_protocol.sh <label>        # 单轮（默认带 GUI）；配置全由 env 决定：
#       KBWALK_GAIT=slow|walk|run    步态，默认 slow：
#           slow = SLOW_WALK 提速到 ~0.6 m/s（已验证的协议三标准剧本）
#           walk = WALK 模式（速度固定模型默认，9/0 键不在白名单无效）
#           run  = RUN 模式 1.5~3.0 m/s ⚠️高动态：obs-lead 未验证，按 dq 边界推断
#                  大概率有害，慎与 SONIC_STATE_OBS_LEAD_S 同开；剧本只前进不倒走
#       SONIC_ENV_PHASE_LOCK / SONIC_STATE_OBS_LEAD_S   照常透传（单轮不强制）
#       JITTER_GUI=0                 单轮想无头时显式传
#
# 按键依据《deploy键盘双键表操作手册》：1/2/3=集0 行走类模式选择；0=+0.1 速度
# （仅 SLOW_WALK/RUN 白名单）；单按 w 全速 23s 自停、连按维持满动量；r=急停。
# ⚠️自动驱动严禁发 n（站立中=IDLE_SQUAT 阶跃当场摔）。
set -u
WT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$WT"

GAIT="${KBWALK_GAIT:-slow}"
case "$GAIT" in slow|walk|run) ;; *) echo "✗ KBWALK_GAIT 只支持 slow|walk|run，当前: $GAIT" >&2; exit 2 ;; esac
if [[ "$GAIT" == "run" && -n "${SONIC_STATE_OBS_LEAD_S:-}" && "${SONIC_STATE_OBS_LEAD_S}" != "0" ]]; then
    echo "⚠️  run 步态(1.5~3.0m/s 高动态) + obs-lead 外推未经验证，摔倒风险高（BVH 判例同 dq 区间）" >&2
fi

drive_keys() {
    local ses="$1"
    DK() { tmux send-keys -t "$ses:deploy" "$1"; }
    case "$GAIT" in
        slow)
            DK '1'; sleep 2
            DK '0'; sleep 1; DK '0'; sleep 1; DK '0'; sleep 1; DK '0'; sleep 2   # 0.2→0.6
            for i in $(seq 1 15); do DK 'w'; sleep 1; done
            DK 'r'; sleep 6
            DK '1'; sleep 1
            for i in $(seq 1 15); do DK 's'; sleep 1; done
            DK 'r'
            ;;
        walk)
            DK '2'; sleep 2
            for i in $(seq 1 15); do DK 'w'; sleep 1; done
            DK 'r'; sleep 6
            DK '2'; sleep 1
            for i in $(seq 1 15); do DK 's'; sleep 1; done
            DK 'r'
            ;;
        run)
            DK '3'; sleep 2
            for i in $(seq 1 10); do DK 'w'; sleep 1; done
            DK 'r'; sleep 8
            DK '3'; sleep 1
            for i in $(seq 1 10); do DK 'w'; sleep 1; done
            DK 'r'
            ;;
    esac
}

run_round() {
    local label="$1"
    while pgrep -f "run_sonic_jitter_closed_loop.sh" >/dev/null 2>&1; do sleep 5; done
    sleep 10
    echo "===== ROUND $label (gait=$GAIT lock=${SONIC_ENV_PHASE_LOCK:-0} tau=${SONIC_STATE_OBS_LEAD_S:-0} gui=${JITTER_GUI:-0}) $(date +%H:%M:%S) ====="
    JITTER_INPUT=keyboard scripts/tools/run_sonic_jitter_closed_loop.sh "$label" -- --free_seconds 150 &
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
    sleep 45   # 锁根 30s + 解锁/稳定余量，进入 free 段
    local ses=$(tmux ls 2>/dev/null | grep -o "sonic_jitter_[0-9]*" | head -1)
    [ -z "$ses" ] && { echo "$label TMUX_NOT_FOUND"; wait $orch; return 1; }
    drive_keys "$ses"
    echo "$label 按键序列完毕 $(date +%H:%M:%S)"
    wait $orch
    echo "$label DONE $(date +%H:%M:%S)"
}

if [ $# -ge 1 ]; then
    # ---- 单轮模式：label 由用户给，配置从当前 env 透传，默认带 GUI ----
    export JITTER_GUI="${JITTER_GUI:-1}"
    run_round "$1"
    exit $?
fi

# ---- 全量 3v3（协议三标准套餐）：lock+τ0 ×3 vs lock+τ10 ×3，A-B-B-A-A-B ----
export JITTER_GUI="${JITTER_GUI:-0}"
export SONIC_ENV_PHASE_LOCK=1
suite_round() { SONIC_STATE_OBS_LEAD_S="$1" run_round "$2"; }
suite_round 0     kwq_a1
suite_round 0.010 kwq_b1
suite_round 0.010 kwq_b2
suite_round 0     kwq_a2
suite_round 0     kwq_a3
suite_round 0.010 kwq_b3
echo "KWQ_3V3_DONE $(date +%H:%M:%S)"
