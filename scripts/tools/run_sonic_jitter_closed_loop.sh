#!/usr/bin/env bash
# SONIC 闭环抖动验证编排器：一条命令跑通 IsaacLab + proxy + deploy 并出抖动指标。
#
# 用法:
#   scripts/tools/run_sonic_jitter_closed_loop.sh <label> [bvh_file] [-- 额外 runner 参数]
# 例:
#   scripts/tools/run_sonic_jitter_closed_loop.sh baseline
#   scripts/tools/run_sonic_jitter_closed_loop.sh fixed /home/nolo/RAYNOS_Motion1.bvh
#   JITTER_GUI=1 scripts/tools/run_sonic_jitter_closed_loop.sh watch -- --free_seconds 60 --hold_seconds 120
#     （带 UI 观察：打开 Isaac 窗口并自动对准机器人；--hold_seconds 让测完后窗口
#       继续实时推进 N 秒供肉眼观察，Ctrl+C 或关窗结束）
#   JITTER_INPUT=keyboard scripts/tools/run_sonic_jitter_closed_loop.sh stand_a
#     （无 BVH 输入端：deploy 走 --input-type keyboard，闭环建立后自动发 Enter
#       启用 Planner，保持 IDLE 静态站立 —— 站立稳定性专用协议，默认
#       locked 30s + free 120s。运动键一概不发。）
#
# 产物: /tmp/sonic_jitter/<label>_<UTC时间>_<pid>/<label>.npz + manifest.json + 日志
# 可用 JITTER_OUT_ROOT 覆盖产物根目录；脚本结束时输出 SONIC_JITTER_RUN_DIR/NPZ。
# 对比: python3 scripts/tools/sonic_jitter_report.py <run-a>/<a>.npz <run-b>/<b>.npz
#
# 链路顺序铁律（KB: project-sonic-bvh-drive-verify）：先启 IsaacLab（sonic_state
# 5560 流动）→ 再启 proxy/deploy → 确认 proxy src=isaac → 才按 ']' 进 CONTROL。
# deploy 在无 lowstate 时进 CONTROL 会输出 NaN last_action（v3 必现，v1 也别赌）。

set -uo pipefail
# 作业控制：让后台 runner 子 shell 自成进程组（组长 pid = $!），cleanup 才能
# kill -- -PID 整组回收。非交互 shell 默认关闭作业控制，后台任务与脚本同组，
# 只杀 $! 会把 isaaclab.sh 之下的 python 泄漏成孤儿（抱死 5560 端口，实测踩过）。
set -m

# 5556/5557/5560/12352 与 tmux 编排都是单实例资源。全局非阻塞锁让两个
# 矩阵/人工运行同时启动时直接失败，不让后启动者清理前一轮的进程。
LOCK_FILE="${SONIC_JITTER_LOCK_FILE:-/tmp/sonic_jitter_closed_loop.lock}"
command -v flock >/dev/null 2>&1 || {
    echo "✗ 缺少 flock，无法保证闭环评测单实例安全" >&2
    exit 2
}
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "✗ 另一轮 SONIC 闭环评测正在运行（锁: $LOCK_FILE）" >&2
    exit 2
fi

ORIGINAL_CLI=("$0" "$@")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ISAACLAB_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "✗ 无法从脚本位置确定 IsaacLab worktree: $SCRIPT_DIR" >&2
    exit 2
}
ISAACLAB_ROOT="$(cd "$ISAACLAB_ROOT" && pwd -P)"
export ISAACLAB_ROOT
export ISAACLAB_PATH="$ISAACLAB_ROOT"

# 强制当前 worktree 的源码包排在 conda editable install/.pth 之前，避免评测实际
# 偷跑另一个 checkout。所有存在的 source 扩展均加入，兼容后续新增扩展包。
SOURCE_PYTHONPATH=()
for extension_dir in "$ISAACLAB_ROOT"/source/*; do
    [[ -d "$extension_dir" ]] && SOURCE_PYTHONPATH+=("$extension_dir")
done
if (( ${#SOURCE_PYTHONPATH[@]} == 0 )); then
    echo "✗ 当前 worktree 下没有 source 包: $ISAACLAB_ROOT/source" >&2
    exit 2
fi
PYTHONPATH_PREFIX="$(IFS=:; printf '%s' "${SOURCE_PYTHONPATH[*]}")"
export PYTHONPATH="${PYTHONPATH_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}"

LABEL="${1:?用法: $0 <label> [bvh_file] [-- runner 参数]}"
shift
if [[ -n "${1:-}" && "${1:-}" != "--" ]]; then
    BVH="$1"; shift
else
    BVH="/home/nolo/RAYNOS_Motion1.bvh"
fi
[[ "${1:-}" == "--" ]] && shift
RUNNER_EXTRA_ARGS=("$@")

# JITTER_INPUT=keyboard：无 BVH 输入端的站立稳定性协议（deploy 键盘 Planner IDLE）。
JITTER_INPUT="${JITTER_INPUT:-bvh}"
case "$JITTER_INPUT" in bvh|keyboard) ;; *) echo "✗ JITTER_INPUT 只支持 bvh|keyboard" >&2; exit 2 ;; esac
if [[ "$JITTER_INPUT" == "keyboard" ]]; then
    # 锁根拉长到 30s：Enter 启用 Planner 落在锁根期内（planner 首次初始化最多 5s），
    # 保证自由根测量段从头到尾都是 Planner IDLE 站立。
    [[ " ${RUNNER_EXTRA_ARGS[*]:-} " == *" --locked_seconds"* ]] || RUNNER_EXTRA_ARGS+=(--locked_seconds 30)
    [[ " ${RUNNER_EXTRA_ARGS[*]:-} " == *" --free_seconds"* ]] || RUNNER_EXTRA_ARGS+=(--free_seconds 120)
fi

# 提取 runner 的显式 seed 写入 manifest。兼容 argparse 的 ``--seed N`` 与
# ``--seed=N`` 两种形式；若重复指定，最后一个值与 argparse 的实际行为一致。
RUN_SEED=""
RUN_SEED_SET=0
for ((arg_index = 0; arg_index < ${#RUNNER_EXTRA_ARGS[@]}; arg_index++)); do
    runner_arg="${RUNNER_EXTRA_ARGS[arg_index]}"
    case "$runner_arg" in
        --seed)
            next_index=$((arg_index + 1))
            if (( next_index >= ${#RUNNER_EXTRA_ARGS[@]} )); then
                echo "✗ runner 参数 --seed 缺少整数值" >&2
                exit 2
            fi
            RUN_SEED="${RUNNER_EXTRA_ARGS[next_index]}"
            RUN_SEED_SET=1
            arg_index=$next_index
            ;;
        --seed=*)
            RUN_SEED="${runner_arg#--seed=}"
            RUN_SEED_SET=1
            ;;
    esac
done
if (( RUN_SEED_SET )) && [[ ! "$RUN_SEED" =~ ^-?[0-9]+$ ]]; then
    echo "✗ runner 参数 --seed 必须是整数，当前为: $RUN_SEED" >&2
    exit 2
fi

SESSION="sonic_jitter_$$"
OUT_ROOT="$(readlink -m -- "${JITTER_OUT_ROOT:-/tmp/sonic_jitter}")"
SONY_REPO_INPUT="${SONY_REPO:-/home/nolo/GR00T-WholeBodyControl-sony-json-stream-20260702}"
SONY_REPO="$(git -C "$SONY_REPO_INPUT" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "✗ SONY_REPO 不是可用 git checkout: $SONY_REPO_INPUT" >&2
    exit 2
}
SONY_REPO="$(cd "$SONY_REPO" && pwd -P)"
# keyboard 模式的 policy 目录（相对 gear_sonic_deploy）。DEPLOY_POLICY_DIR=policy/low_latency
# 切 low-latency 变体（step1 前瞻 ckpt，#9 A/B）；obs config 跟随同目录。
DEPLOY_POLICY_DIR="${DEPLOY_POLICY_DIR:-policy/release}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/miniconda3/envs/env_isaaclab}"
PROTO="${JITTER_POSE_PROTOCOL:-1}"
case "$PROTO" in 1|3) ;; *) echo "✗ JITTER_POSE_PROTOCOL 只支持 1|3，当前为: $PROTO" >&2; exit 2 ;; esac
export SONY_REPO DEPLOY_POLICY_DIR CONDA_ENV_PREFIX
export JITTER_INPUT JITTER_POSE_PROTOCOL="$PROTO"
export JITTER_OUT_ROOT="$OUT_ROOT"

if [[ "$DEPLOY_POLICY_DIR" == /* ]]; then
    POLICY_ROOT="$DEPLOY_POLICY_DIR"
else
    POLICY_ROOT="${SONY_REPO}/gear_sonic_deploy/${DEPLOY_POLICY_DIR}"
fi
DECODER_MODEL="${POLICY_ROOT}/model_decoder.onnx"
ENCODER_MODEL="${POLICY_ROOT}/model_encoder.onnx"
OBS_CONFIG="${POLICY_ROOT}/observation_config.yaml"
PLANNER_MODEL="${SONY_REPO}/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
ROBOT_USD="${SONY_REPO}/gear_sonic/data/robots/g1/g1_43dof.usd"
MANIFEST_HELPER="${ISAACLAB_ROOT}/scripts/tools/sonic_run_manifest.py"
EXTERNAL_LAUNCHER="${SONY_REPO}/scripts/launch_sonic_local_isaaclab_closed_loop.py"
EXTERNAL_WRAPPER="${SONY_REPO}/scripts/launch_sonic_json_isaaclab_closed_loop.sh"
MOCAP_MANAGER="${SONY_REPO}/gear_sonic/scripts/mocap_manager_server.py"
BVH_SENDER="${SONY_REPO}/gear_sonic/scripts/bvh_stream_sender.py"
DEPLOY_ROOT_INPUT="${DEPLOY_ROOT_OVERRIDE:-${SONY_REPO}/gear_sonic_deploy}"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT_INPUT" 2>/dev/null && pwd -P)" || {
    echo "✗ deploy runtime root 不存在或不可访问: $DEPLOY_ROOT_INPUT" >&2
    exit 2
}
DEPLOY_SETUP_ENV="${DEPLOY_ROOT}/scripts/setup_env.sh"
DEPLOY_BIN="${DEPLOY_BIN_OVERRIDE:-${DEPLOY_ROOT}/target/release/g1_deploy_onnx_ref}"
PROXY_BIN="${SONY_REPO}/gear_sonic_deploy/build/tools/sonic_unitree_lowstate_cpp_proxy"
[[ -x "$PROXY_BIN" ]] || PROXY_BIN="${SONY_REPO}/gear_sonic_deploy/prebuilt/linux-x86_64/sonic_unitree_lowstate_cpp_proxy"

SAFE_LABEL="$(printf '%s' "$LABEL" | sed 's/[^[:alnum:]_.-]/_/g')"
[[ -n "$SAFE_LABEL" ]] || SAFE_LABEL="run"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%S_%N)"
RUN_DIR="${OUT_ROOT%/}/${SAFE_LABEL}_${RUN_STAMP}_$$"
ISAAC_LOG="${RUN_DIR}/isaac.log"
OUT_NPZ="${RUN_DIR}/${SAFE_LABEL}.npz"
MANIFEST_JSON="${RUN_DIR}/manifest.json"
RUNNER_STATUS_JSON="${RUN_DIR}/runner_status.json"
TOTAL_TIMEOUT_S=900

# JITTER_GUI=1 = 带 Isaac 窗口跑（观察模式）。headless 去掉；DISPLAY 兜底 :0；
# 首启 shader 编译 + 用户加长观察时间，总超时放宽。
HEADLESS_ARGS=(--headless)
if [[ "${JITTER_GUI:-0}" == "1" ]]; then
    HEADLESS_ARGS=()
    export DISPLAY="${DISPLAY:-:0}"
    TOTAL_TIMEOUT_S=2400
fi
JITTER_GUI="${JITTER_GUI:-0}"
export JITTER_GUI

# 先算出 runner 的最终环境，并把同一组值写入 manifest 与实际子进程。
RUNNER_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "$RUNNER_LD_LIBRARY_PATH" ]]; then
    RUNNER_LD_LIBRARY_PATH="$(
        printf '%s' "$RUNNER_LD_LIBRARY_PATH" |
            tr ':' '\n' |
            grep -v 'cuda-12\.5' |
            paste -sd: || true
    )"
fi
NVJITLINK="${CONDA_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
RUNNER_LD_PRELOAD="${LD_PRELOAD:-}"
if [[ -f "$NVJITLINK" ]]; then
    RUNNER_LD_PRELOAD="${NVJITLINK}${RUNNER_LD_PRELOAD:+:${RUNNER_LD_PRELOAD}}"
fi
RUNNER_ENVIRONMENT=(
    "CONDA_PREFIX=$CONDA_ENV_PREFIX"
    "GR00T_WBC_ROOT=$SONY_REPO"
    "ISAACLAB_PATH=$ISAACLAB_PATH"
    "ISAACLAB_ROOT=$ISAACLAB_ROOT"
    "LD_LIBRARY_PATH=$RUNNER_LD_LIBRARY_PATH"
    "LD_PRELOAD=$RUNNER_LD_PRELOAD"
    "PYTHONPATH=$PYTHONPATH"
    "PYTHONUNBUFFERED=1"
    "TERM=xterm"
    "UNITREE_DDS_DOMAIN_ID=0"
    "UNITREE_DDS_INTERFACE=lo"
    "XR_RUNTIME_JSON=/nonexistent"
)

require_file() {
    local path="$1" desc="$2"
    [[ -f "$path" ]] || { echo "✗ $desc 不存在: $path" >&2; exit 2; }
}
require_executable() {
    local path="$1" desc="$2"
    [[ -x "$path" ]] || { echo "✗ $desc 不可执行: $path" >&2; exit 2; }
}

require_executable "${CONDA_ENV_PREFIX}/bin/python" "IsaacLab conda Python"
require_executable "${ISAACLAB_ROOT}/isaaclab.sh" "isaaclab.sh"
require_file "${ISAACLAB_ROOT}/scripts/tools/sonic_jitter_verify.py" "verify 脚本"
require_file "${ISAACLAB_ROOT}/scripts/tools/sonic_jitter_report.py" "report 脚本"
require_file "$MANIFEST_HELPER" "manifest helper"
require_file "$DECODER_MODEL" "decoder model"
require_file "$ENCODER_MODEL" "encoder model"
require_file "$OBS_CONFIG" "observation config"
require_file "$PLANNER_MODEL" "planner model"
require_file "$ROBOT_USD" "G1 43-DoF robot USD"
require_file "$EXTERNAL_LAUNCHER" "SONY launcher"
require_file "$EXTERNAL_WRAPPER" "SONY launcher wrapper"
require_file "$MOCAP_MANAGER" "mocap manager"
require_file "$BVH_SENDER" "BVH sender"
require_file "$DEPLOY_SETUP_ENV" "deploy runtime setup_env.sh"
require_executable "$DEPLOY_BIN" "deploy binary"
require_executable "$PROXY_BIN" "proxy binary"
DEPLOY_BIN="$(readlink -f -- "$DEPLOY_BIN")"
DEPLOY_RUNTIME_REPO="$(git -C "$DEPLOY_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
DEPLOY_SOURCE="${DEPLOY_ROOT}/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
[[ -f "$DEPLOY_SOURCE" ]] || DEPLOY_SOURCE=""
if [[ -n "$DEPLOY_RUNTIME_REPO" ]]; then
    DEPLOY_RUNTIME_REPO="$(cd "$DEPLOY_RUNTIME_REPO" && pwd -P)"
fi
if [[ "$JITTER_INPUT" == "bvh" ]]; then
    require_file "$BVH" "BVH"
    BVH="$(readlink -f -- "$BVH")"
fi

PORT_PATTERN=':(5557|5560)\b'
if [[ "$JITTER_INPUT" == "bvh" ]]; then
    PORT_PATTERN=':(5556|5557|5560|12352)\b'
fi
PORT_LISTENERS="$(ss -H -ltnp 2>/dev/null | rg "$PORT_PATTERN" || true)"
if [[ -n "$PORT_LISTENERS" ]]; then
    echo "✗ 闭环所需端口已被其他进程监听，拒绝启动：" >&2
    printf '%s\n' "$PORT_LISTENERS" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"
mkdir "$RUN_DIR" || { echo "✗ 无法创建唯一运行目录: $RUN_DIR" >&2; exit 2; }

log() { printf '[jitter-orch %(%H:%M:%S)T] %s\n' -1 "$*"; }

RUNNER_COMMAND=(
    "${ISAACLAB_ROOT}/isaaclab.sh" -p
    "${ISAACLAB_ROOT}/scripts/tools/sonic_jitter_verify.py"
    "${HEADLESS_ARGS[@]}" --device cpu
    --out "$OUT_NPZ"
    --run_manifest "$MANIFEST_JSON"
    --status_file "$RUNNER_STATUS_JSON"
    --kit_args "--/app/vsync=false --/app/runLoops/main/rateLimitEnabled=false"
    "${RUNNER_EXTRA_ARGS[@]}"
)
MANIFEST_COMMAND=(
    "${CONDA_ENV_PREFIX}/bin/python" "$MANIFEST_HELPER" create
    --output "$MANIFEST_JSON"
    --isaaclab-root "$ISAACLAB_ROOT"
    --sony-repo "$SONY_REPO"
    --label "$LABEL"
    --run-dir "$RUN_DIR"
    --out-npz "$OUT_NPZ"
    --isaac-log "$ISAAC_LOG"
    --input "$JITTER_INPUT"
    --pose-protocol "$PROTO"
    --policy-dir "$DEPLOY_POLICY_DIR"
    --policy-root "$POLICY_ROOT"
    --decoder "$DECODER_MODEL"
    --encoder "$ENCODER_MODEL"
    --obs-config "$OBS_CONFIG"
    --planner "$PLANNER_MODEL"
    --robot-usd "$ROBOT_USD"
    --proxy-bin "$PROXY_BIN"
    --deploy-bin "$DEPLOY_BIN"
    --deploy-root "$DEPLOY_ROOT"
    --deploy-setup-env "$DEPLOY_SETUP_ENV"
    --external-launcher "$EXTERNAL_LAUNCHER"
    --external-wrapper "$EXTERNAL_WRAPPER"
    --mocap-manager "$MOCAP_MANAGER"
    --bvh-sender "$BVH_SENDER"
    --session "$SESSION"
)
[[ -n "$DEPLOY_RUNTIME_REPO" ]] && MANIFEST_COMMAND+=(--deploy-runtime-repo "$DEPLOY_RUNTIME_REPO")
[[ -n "$DEPLOY_SOURCE" ]] && MANIFEST_COMMAND+=(--deploy-source "$DEPLOY_SOURCE")
(( RUN_SEED_SET )) && MANIFEST_COMMAND+=(--seed "$RUN_SEED")
[[ "$JITTER_INPUT" == "bvh" ]] && MANIFEST_COMMAND+=(--bvh "$BVH")
[[ "${JITTER_GUI:-0}" == "1" ]] && MANIFEST_COMMAND+=(--gui)
for arg in "${ORIGINAL_CLI[@]}"; do
    MANIFEST_COMMAND+=("--command-arg=$arg")
done
for arg in "${RUNNER_COMMAND[@]}"; do
    MANIFEST_COMMAND+=("--runner-command-arg=$arg")
done
for item in "${RUNNER_ENVIRONMENT[@]}"; do
    MANIFEST_COMMAND+=("--runner-env=$item")
done
for arg in "${RUNNER_EXTRA_ARGS[@]}"; do
    MANIFEST_COMMAND+=("--runner-arg=$arg")
done
"${MANIFEST_COMMAND[@]}" || {
    echo "✗ 无法创建运行 manifest: $MANIFEST_JSON" >&2
    exit 2
}
log "运行目录: $RUN_DIR"
log "manifest: $MANIFEST_JSON"

# ---------- 前置检查：不打别人的进程 ----------
# 不用 pgrep -f：当矩阵 CLI 自身携带 --deploy-bin .../g1_deploy_onnx_ref*
# 时会把父进程命令行误判成 deploy。只检查 /proc/<pid>/exe 的真实可执行文件。
EXTERNAL_DEPLOY_PIDS=()
for process_exe in /proc/[0-9]*/exe; do
    process_target="$(readlink -f -- "$process_exe" 2>/dev/null || true)"
    [[ -n "$process_target" ]] || continue
    if [[ "$(basename -- "$process_target")" == g1_deploy_onnx_ref* ]]; then
        process_pid="${process_exe#/proc/}"
        EXTERNAL_DEPLOY_PIDS+=("${process_pid%/exe}")
    fi
done
if (( ${#EXTERNAL_DEPLOY_PIDS[@]} > 0 )); then
    echo "✗ 检测到本编排器之外的 g1_deploy_onnx_ref 进程在跑，拒绝启动（避免误杀/端口冲突）" >&2
    ps -o pid=,etime=,cmd= -p "$(IFS=,; echo "${EXTERNAL_DEPLOY_PIDS[*]}")" >&2 || true
    exit 2
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "✗ 唯一会话名意外已存在，拒绝接管: $SESSION" >&2
    exit 2
fi

RUNNER_PID=""
OUTPUT_LOCATIONS_PRINTED=0
TEARDOWN_DONE=0
print_output_locations() {
    (( OUTPUT_LOCATIONS_PRINTED == 0 )) || return 0
    printf 'SONIC_JITTER_RUN_DIR=%s\n' "$RUN_DIR"
    printf 'SONIC_JITTER_NPZ=%s\n' "$OUT_NPZ"
    OUTPUT_LOCATIONS_PRINTED=1
}
stop_session_and_wait() {
    (( TEARDOWN_DONE == 0 )) || return 0
    tmux kill-session -t "=$SESSION" 2>/dev/null || true

    # tmux 退出到 ZMQ 监听 socket 真正释放之间存在短暂窗口；矩阵若立刻进入
    # 下一候选会偶发撞上 5557。这里仅等待本轮固定资源释放，不广泛 kill 进程。
    local grace="${SONIC_JITTER_TEARDOWN_GRACE_S:-2}"
    local timeout="${SONIC_JITTER_TEARDOWN_TIMEOUT_S:-30}"
    local waited=0 listeners=""
    sleep "$grace"
    while :; do
        listeners="$(ss -H -ltnp 2>/dev/null | rg "$PORT_PATTERN" || true)"
        if ! tmux has-session -t "=$SESSION" 2>/dev/null && [[ -z "$listeners" ]]; then
            TEARDOWN_DONE=1
            return 0
        fi
        if (( waited >= timeout )); then
            log "✗ teardown ${timeout}s 后资源仍未释放"
            [[ -n "$listeners" ]] && printf '%s\n' "$listeners" >&2
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
}
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
        log "清理: 终止 IsaacLab runner 进程组 (pgid $RUNNER_PID)"
        # RUNNER_PID 是 setsid 出来的组长：负号杀整组，确保 isaaclab.sh 之下的
        # python 子进程一并退出（只杀组长会泄漏 python 抱死 5560 端口）。
        kill -- -"$RUNNER_PID" 2>/dev/null
        sleep 5
        kill -9 -- -"$RUNNER_PID" 2>/dev/null || true
    fi
    if ! stop_session_and_wait && (( status == 0 )); then
        status=1
    fi
    print_output_locations
    exit $status
}
on_signal() {
    local signal_status="$1"
    exit "$signal_status"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

# ---------- 1. 启动 IsaacLab runner（后台） ----------
log "启动 IsaacLab runner → $ISAAC_LOG"
(
    cd "$ISAACLAB_ROOT"
    # nvJitLink 防护（KB: project-nvjitlink-cuda-clash）：使用 manifest 中记录的
    # 最终环境，剔除 CUDA 12.5 并优先预载 conda cu12.8 的 libnvJitLink。
    export LD_LIBRARY_PATH="$RUNNER_LD_LIBRARY_PATH"
    if [[ -n "$RUNNER_LD_PRELOAD" ]]; then
        export LD_PRELOAD="$RUNNER_LD_PRELOAD"
    else
        unset LD_PRELOAD
    fi
    export CONDA_PREFIX="$CONDA_ENV_PREFIX"
    export GR00T_WBC_ROOT="$SONY_REPO"
    export PYTHONUNBUFFERED=1
    export TERM=xterm
    export XR_RUNTIME_JSON=/nonexistent   # headless 严禁挂上 SteamVR（env_hz 崩溃判例）
    export UNITREE_DDS_INTERFACE=lo
    export UNITREE_DDS_DOMAIN_ID=0

    printf '[jitter-import] ISAACLAB_ROOT=%s\n' "$ISAACLAB_ROOT"
    printf '[jitter-import] ISAACLAB_PATH=%s\n' "$ISAACLAB_PATH"
    printf '[jitter-asset] GR00T_WBC_ROOT=%s\n' "$GR00T_WBC_ROOT"
    printf '[jitter-asset] robot_usd=%s\n' "$ROBOT_USD"
    "${CONDA_ENV_PREFIX}/bin/python" "$MANIFEST_HELPER" print-imports \
        --isaaclab-root "$ISAACLAB_ROOT" || exit $?

    exec "${RUNNER_COMMAND[@]}"
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

wait_for_log "$ISAAC_LOG" "waiting for valid deploy targets" 600 "IsaacLab 就绪（5560 状态发布中）"

LAUNCH_STAMP=$(date +%s)
PROXY_LOG=""
DEPLOY_LOG=""

start_explicit_deploy_window() { # start_explicit_deploy_window <keyboard|zmq_manager>
    local input_type="$1"
    local input_args=""
    if [[ "$input_type" == "zmq_manager" ]]; then
        input_args="--zmq-host localhost --zmq-port 5556 --zmq-topic pose"
    fi
    tmux new-window -t "$SESSION" -n deploy \
        "cd '$DEPLOY_ROOT' && export DDS_INTERFACE=lo && source '$DEPLOY_SETUP_ENV' && '$DEPLOY_BIN' \"\$DDS_INTERFACE\" '$DECODER_MODEL' reference/example --obs-config '$OBS_CONFIG' --encoder-file '$ENCODER_MODEL' --planner-file '$PLANNER_MODEL' --input-type '$input_type' $input_args --output-type all --zmq-out-port 5557 --zmq-out-topic g1_debug --disable-crc-check |& tee '$DEPLOY_LOG'; exec bash"
}

wait_deploy_port_free() {
    local waited=0
    while ss -H -ltnp 2>/dev/null | rg -q ':5557\b'; do
        sleep 1
        waited=$((waited + 1))
        if (( waited >= 15 )); then
            log "✗ 本会话旧 deploy 窗口退出后 5557 仍未释放"
            ss -H -ltnp 2>/dev/null | rg ':5557\b' >&2 || true
            return 1
        fi
    done
}

if [[ "$JITTER_INPUT" == "keyboard" ]]; then
    # ---------- 2K. 自建 proxy + deploy(keyboard) 两窗口（无 BVH 输入端） ----------
    # sony launcher 把 --input-type zmq_manager 写死，键盘模式绕开它、
    # 按 launch_sonic_local_isaaclab_closed_loop.py 的 _proxy_command/_deploy_command
    # 原样复刻命令（参数默认值与 BVH 模式完全一致，仅 input-type 不同）。
    PROXY_LOG="${RUN_DIR}/proxy.log"
    DEPLOY_LOG="${RUN_DIR}/deploy.log"
    rm -f "$PROXY_LOG" "$DEPLOY_LOG"
    log "启动 proxy + deploy(--input-type keyboard) (session=$SESSION)"
    tmux new-session -d -s "$SESSION" -n proxy \
        "cd '$SONY_REPO' && export DDS_INTERFACE=lo SDK='${SONY_REPO}/gear_sonic_deploy/thirdparty/unitree_sdk2' && export LD_LIBRARY_PATH=\"\$SDK/thirdparty/lib/\$(uname -m):\$SDK/lib/\$(uname -m):\${LD_LIBRARY_PATH:-}\" && '$PROXY_BIN' --interface \"\$DDS_INTERFACE\" --domain-id 0 --lowstate-hz 500.0 --follow-alpha 0.35 --isaac-state-endpoint tcp://127.0.0.1:5560 --isaac-state-topic sonic_state |& tee '$PROXY_LOG'; exec bash" \
        || { log "✗ tmux proxy 窗口启动失败"; exit 1; }
    start_explicit_deploy_window keyboard \
        || { log "✗ tmux deploy 窗口启动失败"; exit 1; }
elif [[
    "$PROTO" == "1"
    && "$(readlink -f -- "$POLICY_ROOT")" == "$(readlink -f -- "${SONY_REPO}/gear_sonic_deploy/policy/release")"
    && "$DEPLOY_ROOT" == "$(cd "${SONY_REPO}/gear_sonic_deploy" && pwd -P)"
    && "$DEPLOY_BIN" == "$(readlink -f -- "${SONY_REPO}/gear_sonic_deploy/target/release/g1_deploy_onnx_ref")"
]]; then
    # ---------- 2. 启动 sony 三窗口（input/proxy/deploy，默认 v1+release 走原包装层） ----------
    log "启动 sony 闭环三端 (session=$SESSION, bvh=$(basename "$BVH"))"
    (
        cd "$SONY_REPO"
        SESSION="$SESSION" POSE_PROTOCOL_VERSION=1 \
            ./scripts/launch_sonic_json_isaaclab_closed_loop.sh --no-isaaclab --replace "$BVH"
    ) || { log "✗ sony launcher 失败"; exit 1; }
else
    # ---------- 2V. BVH + 非默认协议/policy：直调 python launcher（sony shell 包装层
    # 把 --input-type/policy 路径写死，绕开；参数按 launch_sonic_json_isaaclab_closed_loop.sh
    # 的 launcher_args 原样复刻 + policy 三件套 + 协议版本） ----------
    log "直调 sony python launcher (v$PROTO, policy=$DEPLOY_POLICY_DIR, bvh=$(basename "$BVH"))"
    (
        cd "$SONY_REPO"
        exec ./.venv_teleop/bin/python -u scripts/launch_sonic_local_isaaclab_closed_loop.py \
            --session "$SESSION" --no-attach --replace \
            --repo-root "$SONY_REPO" \
            --proxy-bin "$PROXY_BIN" \
            --no-isaaclab --no-bvh-stream-sender \
            --bvh-stream-port 12352 \
            --bvh-stream-bonedata-coordinate-frame left_handed_yup \
            --bvh-stream-bonedata-position-scale 1.0 \
            --bvh-stream-bonedata-input-quat-order xyzw \
            --bvh-stream-bonedata-rotation-mode input \
            --sony-pico-bonedata-basis zflip \
            --sony-pico-smpl-joints-source pico_fk \
            --pose-protocol-version "$PROTO" \
            --zmq-port 5556 --debug-port 5557 --state-port 5560 \
            --isaac-state-host 127.0.0.1 \
            --decoder "$DECODER_MODEL" \
            --encoder "$ENCODER_MODEL" \
            --obs-config "$OBS_CONFIG"
    ) || { log "✗ sony python launcher 失败"; exit 1; }
    # 外部 launcher 的 deploy recipe 固定执行 SONY_REPO 下 target/release。
    # 评测允许显式 pin 另一 runtime root/build，因此只替换本会话的 deploy
    # 窗口，不改用户文件或生产二进制；proxy/input 仍固定来自 SONY_REPO。
    DEPLOY_LOG="${RUN_DIR}/deploy.log"
    rm -f "$DEPLOY_LOG"
    log "替换本会话 deploy 窗口 → $DEPLOY_BIN"
    tmux kill-window -t "${SESSION}:deploy" 2>/dev/null \
        || { log "✗ 无法关闭本会话默认 deploy 窗口"; exit 1; }
    wait_deploy_port_free || exit 1
    start_explicit_deploy_window zmq_manager \
        || { log "✗ 显式 deploy 窗口启动失败"; exit 1; }
    # BVH 发送端窗口（包装层的 sony_json_sender 等价物，.bvh 走 bvh_stream_sender）
    tmux new-window -t "$SESSION" -n sony_json_sender \
        "cd '$SONY_REPO' && export PYTHONUNBUFFERED=1 PYTHONPATH='$SONY_REPO' && ./.venv_teleop/bin/python -u gear_sonic/scripts/bvh_stream_sender.py --bvh-file '$BVH' --host 127.0.0.1 --port 12352 --fps 50 --unit-scale 0.01 --loop |& tee /tmp/sonic_local_bvh_sender_\$(date +%Y%m%d_%H%M%S).log; exec bash" \
        || { log "✗ BVH 发送端窗口启动失败"; exit 1; }
fi

newest_log() { # newest_log <name>：launch 之后新建的 /tmp/sonic_local_<name>_*.log
    local f
    f=$(ls -t /tmp/sonic_local_"$1"_*.log 2>/dev/null | head -1)
    [[ -n "$f" && $(stat -c %Y "$f") -ge $((LAUNCH_STAMP - 5)) ]] && echo "$f"
}

# ---------- 3. 等 proxy src=isaac（lowstate 链路健康） ----------
waited=0
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
    [[ -z "$DEPLOY_LOG" ]] && DEPLOY_LOG=$(newest_log deploy)
    if [[ -n "$DEPLOY_LOG" ]] && grep -Eq \
        "Unknown encoder observation|Invalid encoder observation|terminate called|Aborted \\(core dumped\\)|Recipe .* failed" \
        "$DEPLOY_LOG" 2>/dev/null; then
        log "✗ deploy 初始化失败；日志尾部："
        tail -30 "$DEPLOY_LOG"
        exit 1
    fi
    tmux send-keys -t "${SESSION}:deploy" ']' 2>/dev/null || true
    sleep 5; waited=$((waited + 5))
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        log "✗ IsaacLab runner 提前退出"; tail -20 "$ISAAC_LOG"; exit 1
    fi
    if (( waited >= 240 )); then
        log "✗ deploy 目标 240s 未进入 IsaacLab；deploy 窗口尾部："
        [[ -z "$DEPLOY_LOG" ]] && DEPLOY_LOG=$(newest_log deploy)
        [[ -n "$DEPLOY_LOG" ]] && tail -25 "$DEPLOY_LOG"
        exit 1
    fi
done
log "✓ 闭环建立，进入测量阶段（锁根跟随 → 解锁自由根）"

if [[ "$JITTER_INPUT" == "keyboard" ]]; then
    # ---------- 4K. 锁根期内启用 Planner（IDLE 静态站立） ----------
    # Enter 是开关键（再按一次切回默认键表），不能像 ']' 那样盲目重发：
    # 发一次后按秒轮询日志最多 12s（planner 首次初始化官方口径 ≤5s），没见回显才重发。
    log "发送 Enter 启用 Planner（保持 IDLE 站立，不发运动键）"
    waited=0
    until grep -q "Planner enabled" "$DEPLOY_LOG" 2>/dev/null; do
        if grep -q "Planner not loaded" "$DEPLOY_LOG" 2>/dev/null; then
            log "✗ deploy 报 Planner not loaded（--planner-file 缺失或路径错）"; exit 1
        fi
        tmux send-keys -t "${SESSION}:deploy" Enter 2>/dev/null || true
        for _ in $(seq 12); do
            sleep 1
            grep -q "Planner enabled" "$DEPLOY_LOG" 2>/dev/null && break
        done
        waited=$((waited + 12))
        if (( waited >= 60 )); then
            log "✗ 60s 未见 Planner enabled；deploy 日志尾部："
            tail -20 "$DEPLOY_LOG"
            exit 1
        fi
    done
    log "✓ Planner enabled（IDLE 站立协议就绪）"
fi

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
if ! stop_session_and_wait; then
    exit 1
fi

VERIFY_STATUS=""
if [[ -f "$RUNNER_STATUS_JSON" ]]; then
    VERIFY_STATUS="$(
        "${CONDA_ENV_PREFIX}/bin/python" -c \
            'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); code=data.get("exit_code"); completed=data.get("completed"); assert isinstance(code, int) and completed is True; print(code)' \
            "$RUNNER_STATUS_JSON" 2>/dev/null
    )" || VERIFY_STATUS=""
fi

if [[ -z "$VERIFY_STATUS" || $RUNNER_STATUS -ne 0 || $VERIFY_STATUS -ne 0 || ! -f "$OUT_NPZ" ]]; then
    log "✗ runner launcher=$RUNNER_STATUS verify=${VERIFY_STATUS:-missing} 或无产物；日志尾部："
    tail -30 "$ISAAC_LOG"
    exit 1
fi

log "✓ 完成: $OUT_NPZ"
REPORT_STATUS=0
"${CONDA_ENV_PREFIX}/bin/python" "$ISAACLAB_ROOT/scripts/tools/sonic_jitter_report.py" "$OUT_NPZ" \
    || REPORT_STATUS=$?
if (( REPORT_STATUS != 0 )); then
    log "✗ report 失败（退出码 $REPORT_STATUS），NPZ 已保留: $OUT_NPZ"
    print_output_locations
    exit "$REPORT_STATUS"
fi
log "✓ report 完成"
print_output_locations
