#!/usr/bin/env bash
# MuJoCo 侧站立对拍编排器：run_sim_loop(弹力带自动放手) + deploy(keyboard Planner IDLE)
# + g1_debug 流录制，产出与 IsaacLab 键盘站立协议同格式的 npz。
#
# 用法:
#   scripts/tools/run_sonic_mujoco_stand_crosscheck.sh <label> [-- 额外 recorder 参数]
# 例:
#   scripts/tools/run_sonic_mujoco_stand_crosscheck.sh mj_stand_1
#   对比: python3 scripts/tools/sonic_jitter_by_group.py /tmp/sonic_jitter/mj_stand_1.npz \
#                 /tmp/sonic_jitter/kb_stand_watch.npz
#
# 协议镜像 IsaacLab 侧（JITTER_INPUT=keyboard）：弹力带吊住 → deploy ']' 进 CONTROL
# → Enter 进 Planner IDLE → 弹力带 sim t=BAND_S 自动放手（≈解锁）→ 放手后录 120s。
# MuJoCo 闭环无 cpp proxy（unitree_sdk2py_bridge 直接发 DDS rt/lowstate）。

set -uo pipefail
set -m

LABEL="${1:?用法: $0 <label> [-- recorder 参数]}"
shift
[[ "${1:-}" == "--" ]] && shift
RECORDER_EXTRA_ARGS=("$@")

SESSION="sonic_mj_crosscheck"
OUT_DIR="/tmp/sonic_jitter"
SONY_REPO="/home/nolo/GR00T-WholeBodyControl-sony-json-stream-20260702"
VENV_PY="${SONY_REPO}/.venv_teleop/bin/python"
MJ_LOG="${OUT_DIR}/mj_sim_${LABEL}.log"
DEPLOY_LOG="${OUT_DIR}/mj_deploy_${LABEL}.log"
REC_LOG="${OUT_DIR}/mj_rec_${LABEL}.log"
OUT_NPZ="${OUT_DIR}/${LABEL}.npz"
DEBUG_PORT=5657
BASE_POSE_PORT=5658
BAND_S="${BAND_S:-90}"          # 弹力带 sim 秒数：deploy 冷启动(774MB planner)要留够
RECORD_S="${RECORD_S:-120}"

mkdir -p "$OUT_DIR"
log() { printf '[mj-orch %(%H:%M:%S)T] %s\n' -1 "$*"; }

# ---------- 前置：不打别人的进程；DDS lo 上只能有一个 deploy ----------
if pgrep -f "g1_deploy_onnx_ref" >/dev/null 2>&1 && ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✗ 检测到本编排器之外的 g1_deploy_onnx_ref，拒绝启动（DDS rt/lowcmd 会冲突）" >&2
    pgrep -af "g1_deploy_onnx_ref" >&2
    exit 2
fi
if pgrep -f "run_sim_loop.py" >/dev/null 2>&1 && ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✗ 检测到已有 run_sim_loop.py 在跑，拒绝启动" >&2
    exit 2
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -f "$MJ_LOG" "$DEPLOY_LOG" "$REC_LOG"

REC_PID=""
cleanup() {
    local status=$?
    if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
        kill -- -"$REC_PID" 2>/dev/null; sleep 2
        kill -9 -- -"$REC_PID" 2>/dev/null || true
    fi
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    exit $status
}
trap cleanup EXIT INT TERM

# ---------- 1. MuJoCo 仿真（弹力带吊住 + base pose 真值 UDP） ----------
log "启动 MuJoCo 仿真 (band=${BAND_S}s, base_pose_udp=${BASE_POSE_PORT})"
tmux new-session -d -s "$SESSION" -n mujoco \
    "cd '$SONY_REPO' && source .venv_sim/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH='$SONY_REPO' DISPLAY=\"\${DISPLAY:-:0}\" SONIC_SIM_BASE_POSE_PORT=$BASE_POSE_PORT SONIC_SIM_BAND_RELEASE_S=$BAND_S && python -u gear_sonic/scripts/run_sim_loop.py |& tee '$MJ_LOG'; exec bash" \
    || { log "✗ tmux mujoco 窗口启动失败"; exit 1; }

wait_for_log() { # wait_for_log <file> <pattern> <timeout_s> <desc>
    local file="$1" pattern="$2" timeout="$3" desc="$4" waited=0
    until grep -q "$pattern" "$file" 2>/dev/null; do
        sleep 5; waited=$((waited + 5))
        if (( waited >= timeout )); then
            log "✗ 等待超时(${timeout}s): $desc"; tail -20 "$file" 2>/dev/null; exit 1
        fi
    done
    log "✓ $desc"
}

wait_for_log "$MJ_LOG" "publishing base pose ground truth" 120 "MuJoCo 仿真起来（base pose 发布中）"
sleep 5

# ---------- 2. deploy（keyboard，g1_debug→5657） ----------
log "启动 deploy (--input-type keyboard, zmq-out $DEBUG_PORT)"
tmux new-window -t "$SESSION" -n deploy \
    "cd '${SONY_REPO}/gear_sonic_deploy' && export DDS_INTERFACE=lo && source scripts/setup_env.sh && just run g1_deploy_onnx_ref \"\$DDS_INTERFACE\" policy/release/model_decoder.onnx reference/example --obs-config policy/release/observation_config.yaml --encoder-file policy/release/model_encoder.onnx --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type keyboard --output-type all --zmq-out-port $DEBUG_PORT --zmq-out-topic g1_debug --disable-crc-check |& tee '$DEPLOY_LOG'; exec bash" \
    || { log "✗ tmux deploy 窗口启动失败"; exit 1; }

# ---------- 3. 反复按 ']' 直到 g1_debug 出包（=CONTROL 进入） ----------
log "向 deploy 发送 ']' 进 CONTROL（以 g1_debug 出包为真值，每 5s 重试）"
probe_debug() {
    "$VENV_PY" - "$DEBUG_PORT" <<'PY'
import sys, zmq
ctx = zmq.Context.instance()
s = ctx.socket(zmq.SUB)
s.setsockopt(zmq.RCVTIMEO, 3000)
s.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
s.connect(f"tcp://127.0.0.1:{sys.argv[1]}")
try:
    s.recv()
    sys.exit(0)
except zmq.Again:
    sys.exit(1)
PY
}
waited=0
until probe_debug; do
    tmux send-keys -t "${SESSION}:deploy" ']' 2>/dev/null || true
    sleep 5; waited=$((waited + 5))
    if (( waited >= 300 )); then
        log "✗ 300s 未见 g1_debug 出包；deploy 日志尾部："; tail -25 "$DEPLOY_LOG"; exit 1
    fi
done
log "✓ CONTROL 已进入（g1_debug 流动）"

# ---------- 4. Enter 进 Planner（IDLE 站立；开关键，不可盲重发） ----------
log "发送 Enter 启用 Planner"
waited=0
until grep -q "Planner enabled" "$DEPLOY_LOG" 2>/dev/null; do
    if grep -q "Planner not loaded" "$DEPLOY_LOG" 2>/dev/null; then
        log "✗ Planner not loaded"; exit 1
    fi
    tmux send-keys -t "${SESSION}:deploy" Enter 2>/dev/null || true
    for _ in $(seq 12); do
        sleep 1
        grep -q "Planner enabled" "$DEPLOY_LOG" 2>/dev/null && break
    done
    waited=$((waited + 12))
    if (( waited >= 60 )); then
        log "✗ 60s 未见 Planner enabled"; tail -20 "$DEPLOY_LOG"; exit 1
    fi
done
log "✓ Planner enabled（IDLE 吊带站立，等弹力带放手）"

# ---------- 5. 等弹力带自动放手 → 立刻起录制器 ----------
wait_for_log "$MJ_LOG" "elastic band auto-released" $((BAND_S + 180)) "弹力带已放手（=解锁，自由站立开始）"
log "启动录制器 ${RECORD_S}s → $OUT_NPZ"
(
    exec "$VENV_PY" -u /home/nolo/xiaoyang_IssacLab/IsaacLab/scripts/tools/sonic_debug_stream_recorder.py \
        --endpoint "tcp://127.0.0.1:${DEBUG_PORT}" --out "$OUT_NPZ" \
        --seconds "$RECORD_S" --base_pose_udp_port "$BASE_POSE_PORT" \
        "${RECORDER_EXTRA_ARGS[@]}"
) &> "$REC_LOG" &
REC_PID=$!

waited=0
while kill -0 "$REC_PID" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    if (( waited >= RECORD_S + 120 )); then
        log "✗ 录制器超时"; tail -10 "$REC_LOG"; exit 1
    fi
done
wait "$REC_PID"; REC_STATUS=$?
REC_PID=""
if [[ $REC_STATUS -ne 0 || ! -f "$OUT_NPZ" ]]; then
    log "✗ 录制器退出码 $REC_STATUS"; tail -15 "$REC_LOG"; exit 1
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true
log "✓ 完成: $OUT_NPZ"
python3 /home/nolo/xiaoyang_IssacLab/IsaacLab/scripts/tools/sonic_jitter_by_group.py "$OUT_NPZ" || true
