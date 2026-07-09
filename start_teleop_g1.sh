#!/usr/bin/env bash
# G1 Locomanipulation XR 遥操启动脚本（Ubuntu 版，对应 Windows 上的 isaaclab.bat 启动方式）
#
# 用法:
#   ./start_teleop_g1.sh                         # 按默认配置启动
#   ./start_teleop_g1.sh --headless ...           # 额外参数原样透传给 teleop_se3_agent.py
#   ./start_teleop_g1.sh --collision-test         # 运行碰撞可视化测试（带 GUI 画面）
#   ./start_teleop_g1.sh --hug-test               # 运行抱箱演示（PD 直驱，证明箱子可被双臂抱住）
#   ./start_teleop_g1.sh --hug-udp-test           # 抱箱演示（伪 deploy 发 UDP，走真实 action manager 链路）
#
# 覆盖默认值（示例）:
#   ISAACLAB_G1_ZMQ_HOST=192.168.50.100 ./start_teleop_g1.sh
set -euo pipefail

# ---------- 可按需修改的配置（均可用同名环境变量在外部覆盖） ----------
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/nolo/xiaoyang_IssacLab/IsaacLab}"

# 碰撞可视化测试 / 抱箱演示模式
_COLLISION_TEST=false
_HUG_TEST=false
_HUG_UDP_TEST=false
_FILTERED_ARGS=()
for _arg in "$@"; do
  if [[ "$_arg" == "--collision-test" ]]; then
    _COLLISION_TEST=true
  elif [[ "$_arg" == "--hug-test" ]]; then
    _HUG_TEST=true
  elif [[ "$_arg" == "--hug-udp-test" ]]; then
    _HUG_UDP_TEST=true
  else
    _FILTERED_ARGS+=("$_arg")
  fi
done
set -- "${_FILTERED_ARGS[@]}"

# G1 43-DoF USD 所在的 GR00T-WholeBodyControl 仓库（import 阶段就要用，缺了直接报错）
export GR00T_WBC_ROOT="${GR00T_WBC_ROOT:-/home/nolo/GR00T-WholeBodyControl}"

# deploy/MuJoCo 发布端所在机器的 IP：本机跑 deploy 用 127.0.0.1，远程则填其局域网 IP
export ISAACLAB_G1_ZMQ_HOST="${ISAACLAB_G1_ZMQ_HOST:-127.0.0.1}"
export ISAACLAB_G1_ROOT_ZMQ_HOST="${ISAACLAB_G1_ROOT_ZMQ_HOST:-$ISAACLAB_G1_ZMQ_HOST}"

# 传输层默认 udp（绑 0.0.0.0:5557/5558 等发送端往本机推）；要走 ZMQ 就取消下一行注释
# export ISAACLAB_G1_TRANSPORT=zmq

TASK="${TASK:-Isaac-PickPlace-Locomanipulation-G1-Abs-v0}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONUNBUFFERED=1

# ---------- conda 环境 ----------
source /home/nolo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

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
# Kit 只通过 XR_RUNTIME_JSON 找 CloudXR runtime（4 个系统级 active_runtime.json 位置全空）。
# 不 source 这个 env 就启动会报 "Cannot start OpenXR! No valid active runtime is set"，
# XR 视口黑屏无响应。详见 KB: NVIDIA/IsaacLab/IsaacSim-StartAR-OpenXR经CloudXR-runtime启动指南.md
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
elif [[ "$_HUG_TEST" == true ]]; then
  echo "[start_teleop_g1] 抱箱演示模式（PD 直驱）"
  exec ./isaaclab.sh -p scripts/environments/teleoperation/hug_box_g1_test.py \
    --device "$DEVICE" \
    --task "$TASK" \
    "$@"
elif [[ "$_HUG_UDP_TEST" == true ]]; then
  echo "[start_teleop_g1] 抱箱演示模式（伪 deploy UDP，走真实 action manager 链路）"
  exec ./isaaclab.sh -p scripts/environments/teleoperation/hug_box_g1_udp_test.py \
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
