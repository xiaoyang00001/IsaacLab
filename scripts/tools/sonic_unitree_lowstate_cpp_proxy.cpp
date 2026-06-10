#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <msgpack.hpp>
#include <zmq.h>

#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using unitree_hg::msg::dds_::IMUState_;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

constexpr int kG1MotorCount = 29;
constexpr int kLowStateMotorCount = 35;

std::atomic<bool> g_running{true};

void handle_signal(int) {
  g_running.store(false);
}

uint32_t monotonic_ms() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

struct Args {
  std::string iface = "enp4s0";
  std::string lowcmd_topic = "rt/lowcmd";
  std::string lowstate_topic = "rt/lowstate";
  std::string imu_topic = "rt/secondary_imu";
  int domain_id = 0;
  int mode_machine = 5;
  double lowstate_hz = 500.0;
  double follow_alpha = 0.35;
  double isaac_state_timeout_s = 0.25;
  std::string isaac_state_endpoint;
  std::string isaac_state_topic = "sonic_state";
};

void print_usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [--interface enp4s0] [--domain-id 0]\n"
      << "       [--lowcmd-topic rt/lowcmd] [--lowstate-topic rt/lowstate]\n"
      << "       [--secondary-imu-topic rt/secondary_imu] [--lowstate-hz 500]\n"
      << "       [--follow-alpha 0.35] [--mode-machine 5]\n"
      << "       [--isaac-state-endpoint tcp://127.0.0.1:5560] [--isaac-state-topic sonic_state]\n"
      << "       [--isaac-state-timeout 0.25]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (key == "--help" || key == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (key == "--interface") {
      const char* value = next();
      if (!value) return false;
      args.iface = value;
    } else if (key == "--domain-id") {
      const char* value = next();
      if (!value) return false;
      args.domain_id = std::stoi(value);
    } else if (key == "--lowcmd-topic") {
      const char* value = next();
      if (!value) return false;
      args.lowcmd_topic = value;
    } else if (key == "--lowstate-topic") {
      const char* value = next();
      if (!value) return false;
      args.lowstate_topic = value;
    } else if (key == "--secondary-imu-topic") {
      const char* value = next();
      if (!value) return false;
      args.imu_topic = value;
    } else if (key == "--lowstate-hz") {
      const char* value = next();
      if (!value) return false;
      args.lowstate_hz = std::stod(value);
    } else if (key == "--follow-alpha") {
      const char* value = next();
      if (!value) return false;
      args.follow_alpha = std::stod(value);
    } else if (key == "--mode-machine") {
      const char* value = next();
      if (!value) return false;
      args.mode_machine = std::stoi(value);
    } else if (key == "--isaac-state-endpoint") {
      const char* value = next();
      if (!value) return false;
      args.isaac_state_endpoint = value;
    } else if (key == "--isaac-state-topic") {
      const char* value = next();
      if (!value) return false;
      args.isaac_state_topic = value;
    } else if (key == "--isaac-state-timeout") {
      const char* value = next();
      if (!value) return false;
      args.isaac_state_timeout_s = std::stod(value);
    } else {
      std::cerr << "Unknown argument: " << key << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  if (args.lowstate_hz <= 0.0) {
    std::cerr << "--lowstate-hz must be positive\n";
    return false;
  }
  args.follow_alpha = std::clamp(args.follow_alpha, 0.0, 1.0);
  return true;
}

std::array<float, kG1MotorCount> default_g1_angles_mujoco() {
  return {
      -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
      -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
      0.0f,    0.0f, 0.0f,
      0.2f,    0.2f, 0.0f, 0.6f,   0.0f,   0.0f, 0.0f,
      0.2f,   -0.2f, 0.0f, 0.6f,   0.0f,   0.0f, 0.0f,
  };
}


struct IsaacState {
  std::array<float, kG1MotorCount> q{};
  std::array<float, kG1MotorCount> dq{};
  std::array<float, kG1MotorCount> ddq{};
  std::array<float, kG1MotorCount> tau{};
  std::array<float, 4> quat{1.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 3> gyro{0.0f, 0.0f, 0.0f};
  std::array<float, 3> accel{0.0f, 0.0f, 9.81f};
  int mode_machine = 5;
  bool valid = false;
  std::chrono::steady_clock::time_point received_at{};
};

template <size_t N>
bool read_float_array(const msgpack::object& obj, std::array<float, N>& dst) {
  if (obj.type != msgpack::type::ARRAY || obj.via.array.size < N) return false;
  for (size_t i = 0; i < N; ++i) {
    try {
      dst[i] = obj.via.array.ptr[i].as<float>();
    } catch (const std::exception&) {
      return false;
    }
  }
  return true;
}

const msgpack::object* find_key(const msgpack::object& map, const char* key) {
  if (map.type != msgpack::type::MAP) return nullptr;
  for (uint32_t i = 0; i < map.via.map.size; ++i) {
    const auto& entry = map.via.map.ptr[i];
    if (entry.key.type != msgpack::type::STR) continue;
    const std::string name(entry.key.via.str.ptr, entry.key.via.str.size);
    if (name == key) return &entry.val;
  }
  return nullptr;
}

class LowStateProxy {
 public:
  explicit LowStateProxy(const Args& args)
      : args_(args),
        target_q_(default_g1_angles_mujoco()),
        sim_q_(default_g1_angles_mujoco()) {
    ChannelFactory::Instance()->Init(args_.domain_id, args_.iface);

    lowstate_pub_ = std::make_unique<ChannelPublisher<LowState_>>(args_.lowstate_topic);
    lowstate_pub_->InitChannel();
    imu_pub_ = std::make_unique<ChannelPublisher<IMUState_>>(args_.imu_topic);
    imu_pub_->InitChannel();
    lowcmd_sub_ = std::make_unique<ChannelSubscriber<LowCmd_>>(args_.lowcmd_topic);
    lowcmd_sub_->InitChannel([this](const void* message) { this->handle_lowcmd(message); }, 1);

    if (!args_.isaac_state_endpoint.empty()) {
      zmq_context_ = zmq_ctx_new();
      zmq_socket_ = zmq_socket(zmq_context_, ZMQ_SUB);
      const int hwm = 1;
      zmq_setsockopt(zmq_socket_, ZMQ_RCVHWM, &hwm, sizeof(hwm));
      zmq_setsockopt(zmq_socket_, ZMQ_SUBSCRIBE, args_.isaac_state_topic.data(), args_.isaac_state_topic.size());
      const int rc = zmq_connect(zmq_socket_, args_.isaac_state_endpoint.c_str());
      if (rc != 0) {
        std::cerr << "[LowStateCppProxy] failed to connect Isaac state endpoint "
                  << args_.isaac_state_endpoint << ": " << zmq_strerror(zmq_errno()) << std::endl;
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
        zmq_ctx_term(zmq_context_);
        zmq_context_ = nullptr;
      }
    }

    lowstate_.mode_machine(static_cast<uint8_t>(args_.mode_machine));
    lowstate_.imu_state().quaternion({1.0f, 0.0f, 0.0f, 0.0f});
    lowstate_.imu_state().gyroscope({0.0f, 0.0f, 0.0f});
    lowstate_.imu_state().accelerometer({0.0f, 0.0f, 9.81f});
    lowstate_.imu_state().rpy({0.0f, 0.0f, 0.0f});

    imu_.quaternion({1.0f, 0.0f, 0.0f, 0.0f});
    imu_.gyroscope({0.0f, 0.0f, 0.0f});
    imu_.accelerometer({0.0f, 0.0f, 9.81f});
    imu_.rpy({0.0f, 0.0f, 0.0f});
  }

  ~LowStateProxy() {
    if (zmq_socket_ != nullptr) zmq_close(zmq_socket_);
    if (zmq_context_ != nullptr) zmq_ctx_term(zmq_context_);
  }

  void run() {
    std::cout << "[LowStateCppProxy] started "
              << "domain=" << args_.domain_id
              << " interface=" << args_.iface
              << " lowcmd=" << args_.lowcmd_topic
              << " lowstate=" << args_.lowstate_topic
              << " imu=" << args_.imu_topic
              << " hz=" << args_.lowstate_hz
              << " alpha=" << args_.follow_alpha
              << " isaac_state=" << (args_.isaac_state_endpoint.empty() ? std::string("<off>") : args_.isaac_state_endpoint)
              << "/" << args_.isaac_state_topic
              << std::endl;

    const auto period = std::chrono::duration<double>(1.0 / args_.lowstate_hz);
    auto next = std::chrono::steady_clock::now();
    auto last = next;
    while (g_running.load()) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= next) {
        const double dt = std::max(std::chrono::duration<double>(now - last).count(), 1.0 / args_.lowstate_hz);
        last = now;
        publish(dt);
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
  }

 private:
  void handle_lowcmd(const void* message) {
    const auto* cmd = static_cast<const LowCmd_*>(message);
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kG1MotorCount; ++i) {
      target_q_[i] = cmd->motor_cmd()[i].q();
    }
    ++lowcmd_count_;
  }

  void drain_isaac_state() {
    if (zmq_socket_ == nullptr) return;
    while (true) {
      char buffer[8192];
      const int size = zmq_recv(zmq_socket_, buffer, sizeof(buffer), ZMQ_DONTWAIT);
      if (size < 0) {
        if (zmq_errno() != EAGAIN) {
          std::cerr << "[LowStateCppProxy] Isaac state recv error: " << zmq_strerror(zmq_errno()) << std::endl;
        }
        return;
      }
      const std::string topic = args_.isaac_state_topic;
      if (size <= static_cast<int>(topic.size()) || std::memcmp(buffer, topic.data(), topic.size()) != 0) {
        continue;
      }
      parse_isaac_state(reinterpret_cast<const char*>(buffer) + topic.size(), size - topic.size());
    }
  }

  void parse_isaac_state(const char* data, size_t size) {
    try {
      msgpack::object_handle handle = msgpack::unpack(data, size);
      const msgpack::object obj = handle.get();
      IsaacState state;
      state.mode_machine = args_.mode_machine;
      const auto* q = find_key(obj, "joint_pos");
      const auto* dq = find_key(obj, "joint_vel");
      const auto* ddq = find_key(obj, "joint_acc");
      const auto* tau = find_key(obj, "joint_tau");
      const auto* quat = find_key(obj, "root_quat_w");
      const auto* gyro = find_key(obj, "root_ang_vel_b");
      const auto* accel = find_key(obj, "root_accel_b");
      const auto* mode = find_key(obj, "mode_machine");
      if (!q || !dq || !read_float_array(*q, state.q) || !read_float_array(*dq, state.dq)) {
        return;
      }
      if (ddq) read_float_array(*ddq, state.ddq);
      if (tau) read_float_array(*tau, state.tau);
      if (quat) read_float_array(*quat, state.quat);
      if (gyro) read_float_array(*gyro, state.gyro);
      if (accel) read_float_array(*accel, state.accel);
      if (mode) state.mode_machine = mode->as<int>();
      state.valid = true;
      state.received_at = std::chrono::steady_clock::now();
      std::lock_guard<std::mutex> lock(mutex_);
      isaac_state_ = state;
      ++isaac_state_count_;
    } catch (const std::exception& exc) {
      std::cerr << "[LowStateCppProxy] failed to parse Isaac state: " << exc.what() << std::endl;
    }
  }

  void publish(double dt) {
    drain_isaac_state();

    std::array<float, kG1MotorCount> target{};
    IsaacState isaac_state;
    uint64_t lowcmd_count = 0;
    uint64_t isaac_state_count = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      target = target_q_;
      lowcmd_count = lowcmd_count_;
      isaac_state = isaac_state_;
      isaac_state_count = isaac_state_count_;
    }

    const auto now = std::chrono::steady_clock::now();
    const bool use_isaac_state = isaac_state.valid
        && std::chrono::duration<double>(now - isaac_state.received_at).count() <= args_.isaac_state_timeout_s;

    if (!use_isaac_state) {
      for (int i = 0; i < kG1MotorCount; ++i) {
        const float prev = sim_q_[i];
        sim_q_[i] = static_cast<float>((1.0 - args_.follow_alpha) * sim_q_[i] + args_.follow_alpha * target[i]);
        sim_dq_[i] = static_cast<float>((sim_q_[i] - prev) / dt);
      }
    }

    lowstate_.tick(monotonic_ms());
    lowstate_.mode_machine(static_cast<uint8_t>(use_isaac_state ? isaac_state.mode_machine : args_.mode_machine));
    for (int i = 0; i < kLowStateMotorCount; ++i) {
      auto& motor = lowstate_.motor_state()[i];
      if (i < kG1MotorCount) {
        motor.q(use_isaac_state ? isaac_state.q[i] : sim_q_[i]);
        motor.dq(use_isaac_state ? isaac_state.dq[i] : sim_dq_[i]);
        motor.ddq(use_isaac_state ? isaac_state.ddq[i] : 0.0f);
        motor.tau_est(use_isaac_state ? isaac_state.tau[i] : 0.0f);
      } else {
        motor.q(0.0f);
        motor.dq(0.0f);
        motor.ddq(0.0f);
        motor.tau_est(0.0f);
      }
    }
    if (use_isaac_state) {
      lowstate_.imu_state().quaternion(isaac_state.quat);
      lowstate_.imu_state().gyroscope(isaac_state.gyro);
      lowstate_.imu_state().accelerometer(isaac_state.accel);
      imu_.quaternion(isaac_state.quat);
      imu_.gyroscope(isaac_state.gyro);
      imu_.accelerometer(isaac_state.accel);
    }

    lowstate_pub_->Write(lowstate_);
    imu_pub_->Write(imu_);
    ++lowstate_count_;

    const auto interval = static_cast<uint64_t>(std::max(args_.lowstate_hz, 1.0));
    if (lowstate_count_ % interval == 0) {
      float absmax = 0.0f;
      for (float q : sim_q_) absmax = std::max(absmax, std::abs(q));
      std::cout << "[LowStateCppProxy] "
                << "lowstate=" << lowstate_count_
                << " lowcmd=" << lowcmd_count
                << " isaac_state=" << isaac_state_count
                << " src=" << (use_isaac_state ? "isaac" : "synthetic")
                << " q_absmax=" << absmax
                << std::endl;
    }
  }

  Args args_;
  LowState_ lowstate_;
  IMUState_ imu_;
  std::unique_ptr<ChannelPublisher<LowState_>> lowstate_pub_;
  std::unique_ptr<ChannelPublisher<IMUState_>> imu_pub_;
  std::unique_ptr<ChannelSubscriber<LowCmd_>> lowcmd_sub_;
  void* zmq_context_ = nullptr;
  void* zmq_socket_ = nullptr;

  std::mutex mutex_;
  std::array<float, kG1MotorCount> target_q_{};
  std::array<float, kG1MotorCount> sim_q_{};
  std::array<float, kG1MotorCount> sim_dq_{};
  IsaacState isaac_state_{};
  uint64_t lowcmd_count_ = 0;
  uint64_t lowstate_count_ = 0;
  uint64_t isaac_state_count_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return 1;
  }
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  LowStateProxy(args).run();
  return 0;
}
