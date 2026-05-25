"""诊断脚本：订阅 ZeroMQ 端口，打印原始收到的数据包，用于排查手柄数据流问题。

用法：
    python scripts/tools/debug_zmq_sub.py
    python scripts/tools/debug_zmq_sub.py --endpoint tcp://192.168.50.105:14025
    python scripts/tools/debug_zmq_sub.py --endpoint tcp://192.168.50.105:14025 --topic state
"""
import argparse
import struct
import sys
import time

try:
    import zmq
except ModuleNotFoundError:
    print("ERROR: pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)

MGXR_MAGIC = 0x4D475852  # 'MGXR'
_HEADER_STRUCT = struct.Struct("<IIIII")

MSG_TYPE_NAMES = {
    0: "PLAYER_ONLINE",
    1: "PLAYER_OFFLINE",
    2: "MOTION_CONTROLLER",
    3: "HEAD_TRACKING",
    4: "HAND_TRACKING",
    5: "WHOLE_BODY",
}


def try_parse_mgxr(data: bytes) -> str:
    if len(data) < _HEADER_STRUCT.size:
        return f"  [too short: {len(data)} bytes]"
    magic, version, player_id, msg_type, payload_size = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != MGXR_MAGIC:
        return f"  [bad magic: 0x{magic:08X}, expected 0x{MGXR_MAGIC:08X}]"
    msg_name = MSG_TYPE_NAMES.get(msg_type, f"UNKNOWN({msg_type})")
    expected_len = _HEADER_STRUCT.size + payload_size
    size_ok = "OK" if len(data) == expected_len else f"MISMATCH(got {len(data)}, expected {expected_len})"
    return f"  MGXR v{version} | player_id={player_id} | type={msg_name} | payload={payload_size}B | size={size_ok}"


def main():
    parser = argparse.ArgumentParser(description="ZeroMQ diagnostic subscriber")
    parser.add_argument("--endpoint", default="tcp://192.168.50.105:14025", help="ZeroMQ endpoint to subscribe to")
    parser.add_argument("--topic", default="", help="Topic filter (empty = all, 'state' = only state topic)")
    parser.add_argument("--timeout", type=int, default=30, help="How many seconds to listen (default 30)")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout per recv
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sock.connect(args.endpoint)

    topic_desc = f'"{args.topic}"' if args.topic else "(all topics)"
    print(f"Subscribing to {args.endpoint} with topic filter {topic_desc}")
    print(f"Listening for {args.timeout} seconds... (Ctrl+C to stop early)\n")

    start = time.time()
    count = 0
    try:
        while time.time() - start < args.timeout:
            try:
                frames = sock.recv_multipart()
            except zmq.Again:
                continue

            count += 1
            print(f"[#{count}] {len(frames)} frame(s) received:")
            for i, frame in enumerate(frames):
                label = "TOPIC" if i == 0 and len(frames) > 1 else "DATA"
                try:
                    text = frame.decode("utf-8", errors="replace")
                    printable = all(32 <= b < 127 or b in (9, 10, 13) for b in frame)
                except Exception:
                    printable = False
                    text = ""

                if i == 0 and len(frames) > 1:
                    print(f"  [{label}] {repr(frame)}")
                else:
                    print(f"  [{label}] {len(frame)} bytes | hex: {frame[:16].hex()}{'...' if len(frame) > 16 else ''}")
                    mgxr_info = try_parse_mgxr(frame)
                    print(mgxr_info)
            print()

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        ctx.term()

    if count == 0:
        print("No messages received.")
        print("Possible causes:")
        print("  1. ZmqServerTest is not publishing to this endpoint")
        print("  2. Topic filter mismatch (try --topic '' to receive all)")
        print("  3. ZmqServerTest socket type is not PUB (e.g. PUSH/DEALER won't work with SUB)")
        print("  4. No VR controller data arriving at ZmqServerTest")
    else:
        print(f"Total: {count} messages received in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
