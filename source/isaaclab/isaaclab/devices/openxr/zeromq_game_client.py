import struct
import threading
from collections import deque
import logging
import time

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

logger = logging.getLogger(__name__)

MGXR_MAGIC = 0x4D475852
MGXR_VERSION = 1

MGXR_MSG_TYPE_PLAYER_ONLINE = 0
MGXR_MSG_TYPE_PLAYER_OFFLINE = 1
MGXR_MSG_TYPE_MOTION_CONTROLLER_TRACKING_INFO = 2
MGXR_MSG_TYPE_HEAD_TRACKING_INFO = 3
MGXR_MSG_TYPE_HAND_TRACKING_INFO = 4
MGXR_MSG_TYPE_WHOLE_BODY_TRACKING_INFO = 5

_HEADER_STRUCT = struct.Struct("<5I")
_POSE_STRUCT = struct.Struct("<7f")
_CONTROLLER_STATES_STRUCT = struct.Struct("<2I3f")

class ZeroMqGameClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ZeroMqGameClient()
        return cls._instance

    def __init__(self):
        self._endpoint = ""
        self._player_id = 0
        self._zmq_context = None
        self._zmq_dealer_socket = None
        self._stop_requested = False
        self._send_queue = deque(maxlen=256)
        self._send_queue_lock = threading.Condition()
        self._send_thread = None
        self._last_send_error_log_ts = 0.0

    def init(self, endpoint="tcp://127.0.0.1:5556", player_id=0):
        if zmq is None:
            logger.error("pyzmq is not installed. ZeroMqGameClient cannot start.")
            return

        if self._send_thread is not None and self._send_thread.is_alive():
            return

        self._endpoint = endpoint
        self._player_id = player_id
        self._zmq_context = zmq.Context()
        self._stop_requested = False

        self._send_thread = threading.Thread(target=self._thread_send_fun, daemon=True)
        self._send_thread.start()

    def _thread_send_fun(self):
        self._zmq_dealer_socket = self._zmq_context.socket(zmq.DEALER)
        self._zmq_dealer_socket.setsockopt(zmq.LINGER, 0)
        self._zmq_dealer_socket.setsockopt(zmq.SNDHWM, 10)
        
        try:
            self._zmq_dealer_socket.connect(self._endpoint)
            logger.info(f"ZeroMqGameClient connected to {self._endpoint}")
        except Exception as e:
            logger.error(f"ZeroMqGameClient connect failed: {e}")
            return

        while not self._stop_requested:
            packet = None
            with self._send_queue_lock:
                self._send_queue_lock.wait_for(lambda: self._stop_requested or len(self._send_queue) > 0)
                if self._stop_requested and len(self._send_queue) == 0:
                    break
                if len(self._send_queue) > 0:
                    packet = self._send_queue.popleft()

            if packet is not None and self._zmq_dealer_socket:
                try:
                    self._zmq_dealer_socket.send(packet, zmq.DONTWAIT)
                except zmq.ZMQError as e:
                    now = time.monotonic()
                    if now - self._last_send_error_log_ts >= 2.0:
                        logger.warning(f"ZeroMqGameClient send failed: {e}")
                        self._last_send_error_log_ts = now

    def enqueue_send_packet(self, packet: bytes):
        with self._send_queue_lock:
            self._send_queue.append(packet)
            self._send_queue_lock.notify_all()

    def send_motion_controller_tracking(self, left_pose, left_inputs, right_pose, right_inputs):
        """
        left_pose/right_pose: [px, py, pz, qw, qx, qy, qz]
        left_inputs/right_inputs: [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding]
        """
        def pack_controller(pose, inputs):
            if pose is None or len(pose) == 0:
                pose = [0.0] * 7
                pose[3] = 1.0 # qw
            if inputs is None or len(inputs) == 0:
                inputs = [0.0] * 7

            px, py, pz, qw, qx, qy, qz = pose
            pose_bytes = _POSE_STRUCT.pack(px, py, pz, qx, qy, qz, qw)

            thumb_x, thumb_y, trigger, squeeze, button_0, button_1, _ = inputs
            buttons = 0
            if button_0 > 0.5: buttons |= (1 << 0)
            if button_1 > 0.5: buttons |= (1 << 1)
            if squeeze > 0.5: buttons |= (1 << 3)
            touches = 0
            inputs_bytes = _CONTROLLER_STATES_STRUCT.pack(buttons, touches, thumb_x, thumb_y, trigger)
            
            return pose_bytes + inputs_bytes

        left_bytes = pack_controller(left_pose, left_inputs)
        right_bytes = pack_controller(right_pose, right_inputs)

        payload_type_bytes = struct.pack("<I", MGXR_MSG_TYPE_MOTION_CONTROLLER_TRACKING_INFO)
        payload = payload_type_bytes + left_bytes + right_bytes
        
        header = _HEADER_STRUCT.pack(
            MGXR_MAGIC,
            MGXR_VERSION,
            self._player_id,
            MGXR_MSG_TYPE_MOTION_CONTROLLER_TRACKING_INFO,
            len(payload)
        )

        self.enqueue_send_packet(header + payload)

    def send_head_tracking(self, head_pose):
        """Send headset tracking pose.

        Args:
            head_pose: ``[px, py, pz, qw, qx, qy, qz]``.
        """
        if head_pose is None or len(head_pose) == 0:
            return

        px, py, pz, qw, qx, qy, qz = head_pose
        pose_bytes = _POSE_STRUCT.pack(px, py, pz, qx, qy, qz, qw)
        payload_type_bytes = struct.pack("<I", MGXR_MSG_TYPE_HEAD_TRACKING_INFO)
        payload = payload_type_bytes + pose_bytes

        header = _HEADER_STRUCT.pack(
            MGXR_MAGIC,
            MGXR_VERSION,
            self._player_id,
            MGXR_MSG_TYPE_HEAD_TRACKING_INFO,
            len(payload),
        )

        self.enqueue_send_packet(header + payload)

    def shutdown(self):
        self._stop_requested = True
        with self._send_queue_lock:
            self._send_queue_lock.notify_all()
        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join()
        if self._zmq_dealer_socket:
            self._zmq_dealer_socket.close()
            self._zmq_dealer_socket = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None
