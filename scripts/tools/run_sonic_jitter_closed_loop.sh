#!/usr/bin/env bash
# SONIC 闭环抖动验证编排器：一条命令跑通 IsaacLab + proxy + deploy(BVH v1 流) 并出抖动指标。
#
# 用法:
#   scripts/tools/run_sonic_jitter_closed_loop.sh <label> [bvh_file] [-- 额外 runner 参数]
# 例:
#   scripts/tools/run_sonic_jitter_closed_loop.sh baseline
#   scripts/tools/run_sonic_jitter_closed_loop.sh fixed /home/nolo/RAYNOS_Motion1.bvh
#   JITTER_GUI=1 scripts/tools/run_sonic_jitter_closed_loop.sh watch -- --free_seconds 60 --hold_seconds 120
#     （带 UI 观察：打开 Isaac 窗口并自动对准机器人；--hold_seconds 让测完后窗口
#       继续实时推进 N 秒供肉眼观察，Ctrl+C 或关窗结束）
#
# 产物: /tmp/sonic_jitter/<label>.npz + isaac_<label>.log
# 对比: python3 scripts/tools/sonic_jitter_report.py /tmp/sonic_jitter/a.npz /tmp/sonic_jitter/b.npz
#
# 链路顺序铁律（KB: project-sonic-bvh-drive-verify）：先启 IsaacLab（sonic_state
# 5560 流动）→ 再启 proxy/deploy → 确认 proxy src=isaac → 才按 ']' 进 CONTROL。
# deploy 在无 lowstate 时进 CONTROL 会输出 NaN last_action（v3 必现，v1 也别赌）。

set -uo pipefail
# 作业控制：让后台 runner 子 shell 自成进程组（组长 pid = $!），cleanup 才能
# kill -- -PID 整组回收。非交互 shell 默认关闭作业控制，后台任务与脚本同组，
# 只杀 $! 会把 isaaclab.sh 之下的 python 泄漏成孤儿（抱死 5560 端口，实测踩过）。
set -m

LABEL="${1:?用法: $0 <label> [bvh_file] [-- runner 参数]}"
shift
if [[ -n "${1:-}" && "${1:-}" != "--" ]]; then
    BVH="$1"; shift
else
    BVH="/home/nolo/RAYNOS_Motion1.bvh"
fi
[[ "${1:-}" == "--" ]] && shift
RUNNER_EXTRA_ARGS=("$@")

SESSION="sonic_jitter"
OUT_DIR="/tmp/sonic_jitter"
ISAACLAB_ROOT="/home/nolo/xiaoyang_IssacLab/IsaacLab"
SONY_REPO="/home/nolo/GR00T-WholeBodyControl-sony-json-stream-20260702"
CONDA_ENV_PREFIX="$HOME/miniconda3/envs/env_isaaclab"
ISAAC_LOG="${OUT_DIR}/isaac_${LABEL}.log"
OUT_NPZ="${OUT_DIR}/${LABEL}.npz"
TOTAL_TIMEOUT_S=900

# JITTER_GUI=1 = 带 Isaac 窗口跑（观察模式）。headless 去掉；DISPLAY 兜底 :0；
# 首启 shader 编译 + 用户加长观察时间，总超时放宽。
HEADLESS_ARGS=(--headless)
if [[ "${JITTER_GUI:-0}" == "1" ]]; then
    HEADLESS_ARGS=()
    export DISPLAY="${DISPLAY:-:0}"
    TOTAL_TIMEOUT_S=2400
fi

mkdir -p "$OUT_DIR"
[[ -f "$BVH" ]] || { echo "✗ BVH 不存在: $BVH" >&2; exit 2; }

log() { printf '[jitter-orch %(%H:%M:%S)T] %s\n' -1 "$*"; }

# ---------- 前置检查：不打别人的进程 ----------
if pgrep -f "g1_deploy_onnx_ref" >/dev/null 2>&1 && ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✗ 检测到本编排器之外的 g1_deploy_onnx_ref 进程在跑，拒绝启动（避免误杀/端口冲突）" >&2
    pgrep -af "g1_deploy_onnx_ref" >&2
    exit 2
fi
tmux kill-session -t "$SESSION" 2>/dev/null || true

RUNNER_PID=""
cleanup() {
    local status=$?
    if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
        log "清理: 终止 IsaacLab runner 进程组 (pgid $RUNNER_PID)"
        # RUNNER_PID 是 setsid 出来的组长：负号杀整组，确保 isaaclab.sh 之下的
        # python 子进程一并退出（只杀组长会泄漏 python 抱死 5560 端口）。
        kill -- -"$RUNNER_PID" 2>/dev/null
        sleep 5
        kill -9 -- -"$RUNNER_PID" 2>/dev/null || true
    fi
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    exit $status
}
trap cleanup EXIT INT TERM

# ---------- 1. 启动 IsaacLab runner（后台） ----------
log "启动 IsaacLab runner → $ISAAC_LOG"
(
    cd "$ISAACLAB_ROOT"
    # nvJitLink 防护（KB: project-nvjitlink-cuda-clash）：剔除 bashrc 的 CUDA12.5
    # 路径并预载 conda env 内 cu12.8 的 libnvJitLink，任何 Kit 启动入口都必须带。
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda-12\.5' | paste -sd: || true)
        export LD_LIBRARY_PATH
    fi
    NVJITLINK="${CONDA_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
    [[ -f "$NVJITLINK" ]] && export LD_PRELOAD="${NVJITLINK}${LD_PRELOAD:+:${LD_PRELOAD}}"

    export CONDA_PREFIX="$CONDA_ENV_PREFIX"
    export PYTHONUNBUFFERED=1
    export XR_RUNTIME_JSON=/nonexistent   # headless 严禁挂上 SteamVR（env_hz 崩溃判例）
    export UNITREE_DDS_INTERFACE=lo
    export UNITREE_DDS_DOMAIN_ID=0

    exec ./isaaclab.sh -p scripts/tools/sonic_jitter_verify.py \
        "${HEADLESS_ARGS[@]}" --device cpu \
        --out "$OUT_NPZ" \
        --kit_args "--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false" \
        "${RUNNER_EXTRA_ARGS[@]}"
) &> "$ISAAC_LOG" &
RUNNER_PID=$!   # set -m 下 = 该后台作业的进程组长

wait_for_log() { # wait_for_log <file> <pattern> <timeout_s> <desc>
    local file="$1" pattern="$2" timeout="$3" desc="$4" waited=0
    while ! grep -q "$pattern" "$file" 2>/dev/null; do
        if ! kill -0 "$RUNNER_PID" 2>/dev/null && [[ "$file" == "$ISAAC_LOG" ]]; then
            log "✗ IsaacLab runner 提前退出，日志尾部："
            tail -20 "$ISAAC_LOG"
            exit 1
        fi
        sleep 5; waited=$((waited + 5))
        if (( waited >= timeout )); then
            log "✗ 等待超时(${timeout}s): $desc"
            tail -20 "$file" 2>/dev/null
            exit 1
        fi
    done
    log "✓ $desc"
}

wait_for_log "$ISAAC_LOG" "waiting for deploy packets" 600 "IsaacLab 就绪（5560 状态发布中）"

# ---------- 2. 启动 sony 三窗口（input/proxy/deploy） ----------
log "启动 sony 闭环三端 (session=$SESSION, bvh=$(basename "$BVH"))"
LAUNCH_STAMP=$(date +%s)
(
    cd "$SONY_REPO"
    SESSION="$SESSION" POSE_PROTOCOL_VERSION=1 \
        ./scripts/launch_sonic_json_isaaclab_closed_loop.sh --no-isaaclab --replace "$BVH"
) || { log "✗ sony launcher 失败"; exit 1; }

newest_log() { # newest_log <name>：launch 之后新建的 /tmp/sonic_local_<name>_*.log
    local f
    f=$(ls -t /tmp/sonic_local_"$1"_*.log 2>/dev/null | head -1)
    [[ -n "$f" && $(stat -c %Y "$f") -ge $((LAUNCH_STAMP - 5)) ]] && echo "$f"
}

# ---------- 3. 等 proxy src=isaac（lowstate 链路健康） ----------
waited=0
PROXY_LOG=""
while :; do
    [[ -z "$PROXY_LOG" ]] && PROXY_LOG=$(newest_log proxy)
    if [[ -n "$PROXY_LOG" ]] && grep -q "src=isaac" "$PROXY_LOG" 2>/dev/null; then
        log "✓ proxy src=isaac ($PROXY_LOG)"
        break
    fi
    sleep 5; waited=$((waited + 5))
    if (( waited >= 240 )); then
        log "✗ proxy 未进入 src=isaac"
        [[ -n "$PROXY_LOG" ]] && tail -20 "$PROXY_LOG"
        exit 1
    fi
done

# ---------- 4. 反复按 ']' 直到 deploy 目标进入 IsaacLab ----------
log "向 deploy 窗口发送 ']' 进 CONTROL（每 5s 重试）"
waited=0
until grep -q "deploy targets flowing" "$ISAAC_LOG" 2>/dev/null; do
    tmux send-keys -t "${SESSION}:deploy" ']' 2>/dev/null || true
    sleep 5; waited=$((waited + 5))
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        log "✗ IsaacLab runner 提前退出"; tail -20 "$ISAAC_LOG"; exit 1
    fi
    if (( waited >= 240 )); then
        log "✗ deploy 目标 240s 未进入 IsaacLab；deploy 窗口尾部："
        DEPLOY_LOG=$(newest_log deploy); [[ -n "$DEPLOY_LOG" ]] && tail -25 "$DEPLOY_LOG"
        exit 1
    fi
done
log "✓ 闭环建立，进入测量阶段（锁根跟随 → 解锁自由根）"

# ---------- 5. 等 runner 完成 ----------
waited=0
while kill -0 "$RUNNER_PID" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    if (( waited >= TOTAL_TIMEOUT_S )); then
        log "✗ 总超时(${TOTAL_TIMEOUT_S}s)，强制结束"
        exit 1
    fi
done
wait "$RUNNER_PID"; RUNNER_STATUS=$?
RUNNER_PID=""
tmux kill-session -t "$SESSION" 2>/dev/null || true

if [[ $RUNNER_STATUS -ne 0 || ! -f "$OUT_NPZ" ]]; then
    log "✗ runner 退出码 $RUNNER_STATUS 或无产物；日志尾部："
    tail -30 "$ISAAC_LOG"
    exit 1
fi

log "✓ 完成: $OUT_NPZ"
python3 "$ISAACLAB_ROOT/scripts/tools/sonic_jitter_report.py" "$OUT_NPZ" || true
