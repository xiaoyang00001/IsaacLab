#!/usr/bin/env bash
# G1 Locomanipulation XR 遥操启动脚本（Ubuntu 版，对应 Windows 上的 isaaclab.bat 启动方式）
#
# 用法:
#   ./start_teleop_g1.sh                         # 按默认配置启动
#   ./start_teleop_g1.sh --headless ...           # 额外参数原样透传给 teleop_se3_agent.py
#   ./start_teleop_g1.sh --collision-test         # 运行碰撞可视化测试（带 GUI 画面）
#
# 覆盖默认值（示例）:
#   ISAACLAB_G1_ZMQ_HOST=192.168.50.100 ./start_teleop_g1.sh
set -euo pipefail

# ---------- 可按需修改的配置（均可用同名环境变量在外部覆盖） ----------
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/nolovr/IsaacLab}"

# 碰撞可视化测试模式
_COLLISION_TEST=false
_FILTERED_ARGS=()
for _arg in "$@"; do
  if [[ "$_arg" == "--collision-test" ]]; then
    _COLLISION_TEST=true
  else
    _FILTERED_ARGS+=("$_arg")
  fi
done
set -- "${_FILTERED_ARGS[@]}"

# G1 43-DoF USD 所在的 GR00T-WholeBodyControl 仓库（import 阶段就要用，缺了直接报错）
export GR00T_WBC_ROOT="${GR00T_WBC_ROOT:-/home/nolovr/GR00T-WholeBodyControl}"

# deploy/MuJoCo 发布端所在机器的 IP：本机跑 deploy 用 127.0.0.1，远程则填其局域网 IP
export ISAACLAB_G1_ZMQ_HOST="${ISAACLAB_G1_ZMQ_HOST:-127.0.0.1}"
export ISAACLAB_G1_ROOT_ZMQ_HOST="${ISAACLAB_G1_ROOT_ZMQ_HOST:-$ISAACLAB_G1_ZMQ_HOST}"

# 传输层默认 udp（绑 0.0.0.0:5557/5558 等发送端往本机推）；要走 ZMQ 就取消下一行注释
# export ISAACLAB_G1_TRANSPORT=zmq

TASK="${TASK:-Isaac-PickPlace-Locomanipulation-G1-Abs-v0}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONUNBUFFERED=1

# ---------- conda 环境 ----------
source /home/nolovr/miniconda3/etc/profile.d/conda.sh
conda activate xiaoyang_isaaclab

# ---------- CUDA 库冲突防护（勿删） ----------
# ~/.bashrc 把系统 CUDA 12.5 塞进了 LD_LIBRARY_PATH，Isaac Sim (Kit) 启动时会先驻留
# 其旧版 libnvJitLink.so.12（只到 12_5 符号），之后 torch(cu128) 的 libcusparse 一加载
# 就报 undefined symbol: __nvJitLinkCreate_12_8。两道防线（仅本次启动生效，不动 bashrc）：
# ① 从 LD_LIBRARY_PATH 剔除 cuda-12.5 相关目录
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH=$(printf '%s' "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda-12\.5' | paste -sd: || true)
  export LD_LIBRARY_PATH
fi
# ② 强制预加载 torch 配套的 nvJitLink 12.8（含 12_0~12_8 全部版本符号，向后兼容）
NVJITLINK="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
[[ -f "$NVJITLINK" ]] && export LD_PRELOAD="$NVJITLINK${LD_PRELOAD:+:$LD_PRELOAD}"

cd "$ISAACLAB_DIR"

# ---------- OpenXR runtime（--xr 必需，勿删） ----------
# 两条 XR 通路，二选一，由 ISAACLAB_XR_RUNTIME 决定（默认 steamvr）：
#
#   steamvr : Isaac Sim → SteamVR(OpenXR runtime) → NOLO driver → PICO 无线串流
#             KB: NVIDIA/CloudXR-OpenXR/NOLO-XRLink-SteamVR-driver部署实战.md
#   cloudxr : Isaac Sim → NVIDIA CloudXR runtime → WebXR/WebRTC → 头显（原通路）
#             KB: NVIDIA/IsaacLab/IsaacSim-StartAR-OpenXR经CloudXR-runtime启动指南.md
#
# Kit 只在**进程启动时**读一次 XR_RUNTIME_JSON；该环境变量的优先级高于系统的
# ~/.config/openxr/1/active_runtime.json。不设或设错都会报
# "Cannot start OpenXR! No valid active runtime is set"，XR 视口黑屏无响应。
XR_BACKEND="${ISAACLAB_XR_RUNTIME:-steamvr}"

case "$XR_BACKEND" in
  steamvr)
    STEAMVR_XR_JSON="$HOME/.steam/steam/steamapps/common/SteamVR/steamxr_linux64.json"
    if [[ -f "$STEAMVR_XR_JSON" ]]; then
      export XR_RUNTIME_JSON="$STEAMVR_XR_JSON"
    else
      echo "[start_teleop_g1] ⚠️ 未找到 SteamVR 的 OpenXR manifest：$STEAMVR_XR_JSON" >&2
      echo "[start_teleop_g1]    SteamVR 装了吗？--xr 会黑屏。" >&2
    fi
    # SteamVR 必须已在运行：vrclient.so 要连 vrserver，没起来 OpenXR 会话建不起来
    if ! pgrep -x vrserver >/dev/null 2>&1; then
      echo "[start_teleop_g1] ⚠️ SteamVR(vrserver) 未运行 —— --xr 会黑屏。先启动它：" >&2
      echo "[start_teleop_g1]    env -u http_proxy -u https_proxy -u all_proxy DISPLAY=:0 setsid /usr/games/steam steam://run/250820" >&2
      echo "[start_teleop_g1]    （必须走 steam:// 让它跑在 sniper 容器里，裸跑 vrstartup 会崩）" >&2
    fi
    ;;

  cloudxr)
    CLOUDXR_ENV="$HOME/.cloudxr/run/cloudxr.env"
    if [[ -f "$CLOUDXR_ENV" ]]; then
      # shellcheck disable=SC1090
      source "$CLOUDXR_ENV"
    else
      echo "[start_teleop_g1] ⚠️ 未找到 $CLOUDXR_ENV，CloudXR runtime 可能没启动，--xr 会黑屏" >&2
    fi
    if [[ ! -S "$HOME/.cloudxr/run/ipc_cloudxr" ]]; then
      echo "[start_teleop_g1] ⚠️ CloudXR runtime 的 ipc socket 不存在，请先启动 runtime（isaacteleop.cloudxr.runtime）" >&2
    fi
    ;;

  *)
    echo "[start_teleop_g1] ✗ ISAACLAB_XR_RUNTIME 只能是 steamvr 或 cloudxr，当前：$XR_BACKEND" >&2
    exit 1
    ;;
esac

echo "[start_teleop_g1] XR 通路           = $XR_BACKEND$([[ "$XR_BACKEND" == steamvr ]] && echo '（默认；切回 CloudXR: ISAACLAB_XR_RUNTIME=cloudxr）')"
echo "[start_teleop_g1] XR_RUNTIME_JSON    = ${XR_RUNTIME_JSON:-<未设置>}"
echo "[start_teleop_g1] GR00T_WBC_ROOT     = $GR00T_WBC_ROOT"
echo "[start_teleop_g1] ZMQ_HOST           = $ISAACLAB_G1_ZMQ_HOST (root: $ISAACLAB_G1_ROOT_ZMQ_HOST)"
echo "[start_teleop_g1] TRANSPORT          = ${ISAACLAB_G1_TRANSPORT:-udp(默认)}"
echo "[start_teleop_g1] TASK / DEVICE      = $TASK / $DEVICE"
echo "[start_teleop_g1] 本机 IP（PICO 串流填这个）: $(hostname -I | awk '{print $1}')"

if [[ "$_COLLISION_TEST" == true ]]; then
  echo "[start_teleop_g1] 碰撞可视化测试模式"
  exec ./isaaclab.sh -p scripts/environments/teleoperation/collision_g1_test.py \
    --device "$DEVICE" \
    --task "$TASK" \
    "$@"
else
  exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --device "$DEVICE" \
    --task "$TASK" \
    --teleop_device motion_controllers \
    --enable_pinocchio \
    "$@"
fi
