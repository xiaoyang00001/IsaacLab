"""模拟 VR 手柄向 ZmqServerTest 发送 MGXR 数据包，用于测试手柄数据流。

ZmqServerTest 架构：
  DEALER (本脚本) ──► ROUTER:14026 ──► PUB:14025 (topic="state") ──► IsaacLab ZeroMqGameSubDevice

用法：
    python scripts/tools/simulate_vr_controller.py
    python scripts/tools/simulate_vr_controller.py --endpoint tcp://192.168.50.105:14026 --player_id 2
    python scripts/tools/simulate_vr_controller.py --send_start    # 先发一帧 button_0 触发 START
    python scripts/tools/simulate_vr_controller.py --mode circle   # 手腕画圆
    python scripts/tools/simulate_vr_controller.py --mode still    # 保持静止中性姿态
"""
import argparse
import math
import struct
import sys
import time

try:
    import zmq
except ModuleNotFoundError:
    print("ERROR: pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)

# ── MGXR 协议常量 ─────────────────────────────────────────────────────────────
MGXR_MAGIC = 0x4D475852
MGXR_VERSION = 1

MSG_PLAYER_ONLINE = 0
MSG_PLAYER_OFFLINE = 1
MSG_MOTION_CONTROLLER = 2
MSG_HEAD_TRACKING = 3
MSG_HAND_TRACKING = 4
MSG_WHOLE_BODY = 5

# 按照 ZeroMqGameClient 的 pack 格式
_HEADER = struct.Struct("<IIIII")          # magic version player_id msg_type payload_size
_POSE = struct.Struct("<fffffff")          # px py pz  qx qy qz qw  (xyzw on wire)
_CTRL_STATES = struct.Struct("<IIfff")     # buttons touches thumb_x thumb_y trigger

# button 掩码（与 ZeroMqGameSubDeviceCfg 默认值一致）
BTN_0_MASK = 1 << 0   # left→START, right→RESET
BTN_1_MASK = 1 << 1   # left→STOP
SQUEEZE_MASK = 1 << 3


def build_mgxr_packet(player_id: int, msg_type: int, payload: bytes) -> bytes:
    header = _HEADER.pack(MGXR_MAGIC, MGXR_VERSION, player_id, msg_type, len(payload))
    return header + payload


def build_controller_packet(
    player_id: int,
    left_pose: list[float],   # [px, py, pz, qw, qx, qy, qz]  Isaac Lab wxyz
    left_inputs: list[float], # [thumb_x, thumb_y, trigger, squeeze, btn0, btn1, _]
    right_pose: list[float],
    right_inputs: list[float],
) -> bytes:
    """打包一帧 MOTION_CONTROLLER_TRACKING_INFO。"""

    def pack_side(pose, inputs):
        px, py, pz, qw, qx, qy, qz = pose          # wxyz → xyzw on wire
        pose_bytes = _POSE.pack(px, py, pz, qx, qy, qz, qw)

        thumb_x, thumb_y, trigger, squeeze, btn0, btn1, _ = inputs
        buttons = 0
        if btn0 > 0.5:    buttons |= BTN_0_MASK
        if btn1 > 0.5:    buttons |= BTN_1_MASK
        if squeeze > 0.5: buttons |= SQUEEZE_MASK
        state_bytes = _CTRL_STATES.pack(buttons, 0, thumb_x, thumb_y, trigger)
        return pose_bytes + state_bytes

    payload_type = struct.pack("<I", MSG_MOTION_CONTROLLER)
    payload = payload_type + pack_side(left_pose, left_inputs) + pack_side(right_pose, right_inputs)
    return build_mgxr_packet(player_id, MSG_MOTION_CONTROLLER, payload)


def build_head_packet(player_id: int, pose: list[float]) -> bytes:
    px, py, pz, qw, qx, qy, qz = pose
    payload_type = struct.pack("<I", MSG_HEAD_TRACKING)
    payload = payload_type + _POSE.pack(px, py, pz, qx, qy, qz, qw)
    return build_mgxr_packet(player_id, MSG_HEAD_TRACKING, payload)


# ── 动画生成器 ────────────────────────────────────────────────────────────────

def neutral_pose() -> list[float]:
    """中性手腕姿态：略微前伸，单位四元数。"""
    return [0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]  # wxyz


def neutral_inputs() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def circle_pose(t: float, radius: float, x_base: float, y_base: float, z_base: float) -> list[float]:
    """在 XZ 平面画圆。"""
    x = x_base + radius * math.cos(t)
    z = z_base + radius * math.sin(t)
    return [x, y_base, z, 1.0, 0.0, 0.0, 0.0]


def squeeze_inputs(t: float) -> list[float]:
    """trigger 随时间缓慢开合。"""
    trigger = max(0.0, 0.5 * (1.0 + math.sin(t * 0.8)))
    return [0.0, 0.0, trigger, 0.0, 0.0, 0.0, 0.0]


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="模拟 VR 手柄向 ZmqServerTest 发送 MGXR 数据")
    parser.add_argument("--endpoint", default="tcp://192.168.50.105:14026",
                        help="ZmqServerTest ROUTER 端口 (default: tcp://192.168.50.105:14026)")
    parser.add_argument("--player_id", type=int, default=2,
                        help="MGXR player_id，必须与 target_remote_player_id 一致 (default: 2)")
    parser.add_argument("--hz", type=float, default=30.0,
                        help="发送频率 Hz (default: 30)")
    parser.add_argument("--mode", choices=["circle", "still"], default="circle",
                        help="动画模式：circle=手腕画圆, still=保持静止 (default: circle)")
    parser.add_argument("--send_start", action="store_true",
                        help="开始时发 1 帧 左手柄 button_0=1 来触发 IsaacLab START 回调")
    parser.add_argument("--radius", type=float, default=0.08,
                        help="circle 模式下圆的半径（米，default: 0.08）")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, 10)
    sock.connect(args.endpoint)

    print(f"连接到 {args.endpoint}，player_id={args.player_id}，模式={args.mode}，{args.hz} Hz")
    print("按 Ctrl+C 停止\n")

    period = 1.0 / args.hz
    t = 0.0

    # ── 发送 START 按钮帧 ──
    if args.send_start:
        start_inputs = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # btn0=1 → START
        pkt = build_controller_packet(
            args.player_id,
            neutral_pose(), start_inputs,
            neutral_pose(), neutral_inputs(),
        )
        sock.send(pkt, zmq.DONTWAIT)
        print("[START] 已发送左手柄 button_0=1 触发 START")
        time.sleep(0.1)

    count = 0
    try:
        while True:
            loop_start = time.monotonic()

            if args.mode == "circle":
                left_pose  = circle_pose(t,  args.radius, -0.3, 0.5,  0.15)
                right_pose = circle_pose(-t, args.radius,  0.3, 0.5,  0.15)
                left_inp   = squeeze_inputs(t)
                right_inp  = squeeze_inputs(t + math.pi)
            else:  # still
                left_pose  = neutral_pose()
                right_pose = neutral_pose()
                left_inp   = neutral_inputs()
                right_inp  = neutral_inputs()

            head_pose = [0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0]

            pkt_ctrl = build_controller_packet(
                args.player_id, left_pose, left_inp, right_pose, right_inp
            )
            pkt_head = build_head_packet(args.player_id, head_pose)

            try:
                sock.send(pkt_head, zmq.DONTWAIT)
                sock.send(pkt_ctrl, zmq.DONTWAIT)
            except zmq.ZMQError as e:
                print(f"发送失败: {e}")

            count += 1
            if count % (int(args.hz) * 2) == 0:
                print(f"[{count // int(args.hz):4d}s] 已发送 {count} 帧 | "
                      f"left=({left_pose[0]:.3f}, {left_pose[1]:.3f}, {left_pose[2]:.3f}) | "
                      f"trigger_L={left_inp[2]:.2f}")

            t += period * 2 * math.pi * 0.3  # 0.3 Hz 转速

            elapsed = time.monotonic() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n停止。共发送 {count} 帧。")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
