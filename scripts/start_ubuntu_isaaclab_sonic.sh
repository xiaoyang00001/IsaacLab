#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
DEFAULT_ISAACLAB_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

die() {
    echo "[sonic-ubuntu-isaaclab] ERROR: $*" >&2
    exit 1
}

log() {
    echo "[sonic-ubuntu-isaaclab] $*"
}

usage() {
    cat <<'EOF'
Start Ubuntu IsaacLab for the SONIC deploy bridge.

Examples:
  cd /path/to/IsaacLab
  ./scripts/start_ubuntu_isaaclab_sonic.sh

  ./scripts/start_ubuntu_isaaclab_sonic.sh \
    --deploy-ip 192.168.50.68 \
    --local-ip 192.168.50.105 \
    --xr \
    --xr-view first

Options:
  --deploy-ip IP             Host running SONIC deploy debug ZMQ. Default: 127.0.0.1.
  --local-ip IP              Local IsaacLab machine IPv4. Default: auto-detected IPv4.
  --machine-a-ip IP          Override ISAACLAB_MACHINE_A_IP.
                             Default: --local-ip, or --deploy-ip when --deploy-ip is set.
  --machine-b-ip IP          Override ISAACLAB_MACHINE_B_IP. Default: --local-ip.
  --tracking-hub-ip IP       Override ISAACLAB_TRACKING_HUB_IP.
                             Default: same as --machine-a-ip.
  --isaaclab-root PATH       IsaacLab root. Default: parent of this script.
  --task TASK                IsaacLab task id.
                             Default: Isaac-SonicSolo-Locomanipulation-G1-v0.
  --device DEVICE            IsaacLab device. Default: cpu.
  --debug-port PORT          SONIC deploy debug ZMQ port. Default: 5557.
  --state-port PORT          IsaacLab state publisher ZMQ port. Default: 5560.
  --deploy-topic TOPIC       SONIC deploy topic. Default: g1_debug.
  --state-topic TOPIC        IsaacLab state topic. Default: sonic_state.
  --physics-mode 0|1         SONIC_G1_PHYSICS_MODE. Default: 1.
  --visual-servo-mode 0|1    SONIC_G1_VISUAL_SERVO_MODE. Default: 0.
  --self-collisions 0|1      SONIC_G1_SELF_COLLISIONS. Default: 0.
  --stabilize-root 0|1       SONIC_DEPLOY_STABILIZE_ROOT. Default: 1.
  --target-rate-limit VALUE  SONIC_DEPLOY_TARGET_RATE_LIMIT. Default: 0.04.
  --headless                 Pass --headless to IsaacLab.
  --xr                       Pass --xr and --teleop_device handtracking to IsaacLab.
  --xr-view first|third      SONIC_XR_VIEW. Default: first.
  --enable-pinocchio         Pass --enable_pinocchio to IsaacLab.
  --kit-arg ARG              Add one Kit arg. Can be repeated.
  --dry-run                  Print resolved environment and command, then exit.
  -h, --help                 Show this help.

Any arguments after "--" are appended to the IsaacLab Python script command.
EOF
}

need_value() {
    local option="$1"
    local value="${2-}"
    [[ -n "${value}" ]] || die "${option} requires a value"
}

validate_01() {
    local name="$1"
    local value="$2"
    [[ "${value}" == "0" || "${value}" == "1" ]] || die "${name} must be 0 or 1"
}

validate_port() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an integer port"
    (( value >= 1 && value <= 65535 )) || die "${name} must be between 1 and 65535"
}

detect_default_ipv4() {
    local detected=""

    if command -v ip >/dev/null 2>&1; then
        detected="$(ip route get 1.1.1.1 2>/dev/null | awk '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i == "src") {
                        print $(i + 1)
                        exit
                    }
                }
            }
        ' || true)"
    fi

    if [[ -z "${detected}" ]] && command -v hostname >/dev/null 2>&1; then
        detected="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    fi

    [[ -n "${detected}" ]] && echo "${detected}"
}

print_command() {
    local -a command=("$@")
    printf '[sonic-ubuntu-isaaclab] Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
}

deploy_ip=""
local_ip=""
machine_a_ip=""
machine_b_ip=""
tracking_hub_ip=""
deploy_ip_explicit=0
isaaclab_root="${DEFAULT_ISAACLAB_ROOT}"

task="Isaac-SonicSolo-Locomanipulation-G1-v0"
device="cpu"
debug_port="5557"
state_port="5560"
deploy_topic="g1_debug"
state_topic="sonic_state"
physics_mode="1"
visual_servo_mode="0"
self_collisions="0"
stabilize_root="1"
target_rate_limit="0.04"
xr_view="first"

headless=0
xr=0
enable_pinocchio=0
dry_run=0

kit_args=(
    "--/app/vsync=false"
    "--/app/runLoops/main/rateLimitEnabled=false"
)
extra_isaac_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy-ip)
            need_value "$1" "${2-}"
            deploy_ip="$2"
            deploy_ip_explicit=1
            shift 2
            ;;
        --local-ip)
            need_value "$1" "${2-}"
            local_ip="$2"
            shift 2
            ;;
        --machine-a-ip)
            need_value "$1" "${2-}"
            machine_a_ip="$2"
            shift 2
            ;;
        --machine-b-ip)
            need_value "$1" "${2-}"
            machine_b_ip="$2"
            shift 2
            ;;
        --tracking-hub-ip)
            need_value "$1" "${2-}"
            tracking_hub_ip="$2"
            shift 2
            ;;
        --isaaclab-root)
            need_value "$1" "${2-}"
            isaaclab_root="$2"
            shift 2
            ;;
        --task)
            need_value "$1" "${2-}"
            task="$2"
            shift 2
            ;;
        --device)
            need_value "$1" "${2-}"
            device="$2"
            shift 2
            ;;
        --debug-port)
            need_value "$1" "${2-}"
            debug_port="$2"
            shift 2
            ;;
        --state-port)
            need_value "$1" "${2-}"
            state_port="$2"
            shift 2
            ;;
        --deploy-topic)
            need_value "$1" "${2-}"
            deploy_topic="$2"
            shift 2
            ;;
        --state-topic)
            need_value "$1" "${2-}"
            state_topic="$2"
            shift 2
            ;;
        --physics-mode)
            need_value "$1" "${2-}"
            physics_mode="$2"
            shift 2
            ;;
        --visual-servo-mode)
            need_value "$1" "${2-}"
            visual_servo_mode="$2"
            shift 2
            ;;
        --self-collisions)
            need_value "$1" "${2-}"
            self_collisions="$2"
            shift 2
            ;;
        --stabilize-root)
            need_value "$1" "${2-}"
            stabilize_root="$2"
            shift 2
            ;;
        --target-rate-limit)
            need_value "$1" "${2-}"
            target_rate_limit="$2"
            shift 2
            ;;
        --headless)
            headless=1
            shift
            ;;
        --xr)
            xr=1
            shift
            ;;
        --xr-view)
            need_value "$1" "${2-}"
            xr_view="$2"
            shift 2
            ;;
        --enable-pinocchio)
            enable_pinocchio=1
            shift
            ;;
        --kit-arg)
            need_value "$1" "${2-}"
            kit_args+=("$2")
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_isaac_args+=("$@")
            break
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

validate_port "--debug-port" "${debug_port}"
validate_port "--state-port" "${state_port}"
validate_01 "--physics-mode" "${physics_mode}"
validate_01 "--visual-servo-mode" "${visual_servo_mode}"
validate_01 "--self-collisions" "${self_collisions}"
validate_01 "--stabilize-root" "${stabilize_root}"

case "${xr_view}" in
    first|third) ;;
    *) die "--xr-view must be first or third" ;;
esac

detected_ip="$(detect_default_ipv4 || true)"
local_ip="${local_ip:-${detected_ip:-127.0.0.1}}"
deploy_ip="${deploy_ip:-127.0.0.1}"

if (( deploy_ip_explicit )); then
    default_machine_a_ip="${deploy_ip}"
else
    default_machine_a_ip="${local_ip}"
fi

machine_a_ip="${machine_a_ip:-${default_machine_a_ip}}"
machine_b_ip="${machine_b_ip:-${local_ip}}"
tracking_hub_ip="${tracking_hub_ip:-${machine_a_ip}}"

[[ -d "${isaaclab_root}" ]] || die "IsaacLab root does not exist: ${isaaclab_root}"
isaaclab_root="$(cd -- "${isaaclab_root}" >/dev/null 2>&1 && pwd -P)"
isaaclab_sh="${isaaclab_root}/isaaclab.sh"
[[ -f "${isaaclab_sh}" ]] || die "isaaclab.sh not found under IsaacLab root: ${isaaclab_root}"
[[ -x "${isaaclab_sh}" ]] || die "isaaclab.sh is not executable: ${isaaclab_sh}"

export ISAACLAB_MACHINE_A_IP="${machine_a_ip}"
export ISAACLAB_MACHINE_B_IP="${machine_b_ip}"
export ISAACLAB_LOCAL_MACHINE_IP="${local_ip}"
export ISAACLAB_TRACKING_HUB_IP="${tracking_hub_ip}"

export SONIC_DEPLOY_TRANSPORT="zmq"
export SONIC_DEPLOY_ENDPOINT="tcp://${deploy_ip}:${debug_port}"
export SONIC_DEPLOY_TOPIC="${deploy_topic}"
export SONIC_DEPLOY_TARGET_FIELD="last_action"
export SONIC_DEPLOY_REFERENCE_TARGET_FIELD="body_q_target"

export SONIC_PUBLISH_STATE_ZMQ="1"
export SONIC_STATE_ZMQ_BIND="tcp://*:${state_port}"
export SONIC_STATE_ZMQ_TOPIC="${state_topic}"

export SONIC_XR_VIEW="${xr_view}"

export SONIC_G1_PHYSICS_MODE="${physics_mode}"
export SONIC_G1_VISUAL_SERVO_MODE="${visual_servo_mode}"
export SONIC_G1_SELF_COLLISIONS="${self_collisions}"
export SONIC_DEPLOY_STABILIZE_ROOT="${stabilize_root}"
export SONIC_DEPLOY_TARGET_RATE_LIMIT="${target_rate_limit}"

isaac_args=(
    "-p"
    "scripts/environments/teleoperation/teleop_se3_agent.py"
    "--task"
    "${task}"
    "--device"
    "${device}"
    "--kit_args"
    "${kit_args[*]}"
)

if (( headless )); then
    isaac_args+=("--headless")
fi

if (( xr )); then
    isaac_args+=("--xr")
    isaac_args+=("--teleop_device" "handtracking")
fi

if (( enable_pinocchio )); then
    isaac_args+=("--enable_pinocchio")
fi

if (( ${#extra_isaac_args[@]} )); then
    isaac_args+=("${extra_isaac_args[@]}")
fi

log "IsaacLabRoot: ${isaaclab_root}"
log "DeployIp: ${deploy_ip}"
log "LocalIp: ${local_ip}"
log "MachineA: ${ISAACLAB_MACHINE_A_IP}, MachineB: ${ISAACLAB_MACHINE_B_IP}, TrackingHub: ${ISAACLAB_TRACKING_HUB_IP}"
log "Deploy endpoint: ${SONIC_DEPLOY_ENDPOINT}, topic: ${SONIC_DEPLOY_TOPIC}"
log "State bind: ${SONIC_STATE_ZMQ_BIND}, topic: ${SONIC_STATE_ZMQ_TOPIC}"
log "Task: ${task}, device: ${device}, xr: ${xr}, xr view: ${xr_view}"
print_command "./isaaclab.sh" "${isaac_args[@]}"

if (( dry_run )); then
    exit 0
fi

cd -- "${isaaclab_root}"
exec ./isaaclab.sh "${isaac_args[@]}"
