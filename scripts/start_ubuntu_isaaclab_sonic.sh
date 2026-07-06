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
  --auto-recover 0|1         SONIC_DEPLOY_AUTO_RECOVER. Auto fall recovery,
                             matching the MuJoCo reference sim: root height
                             below 0.2 m stands the robot back up in place and
                             re-unlocks after settle. Set 0 for manual-only
                             (J key stands up, U/START unlocks). Default: 1.
  --target-rate-limit VALUE  SONIC_DEPLOY_TARGET_RATE_LIMIT. Default: 0.04.
  --headless                 Pass --headless to IsaacLab.
  --xr                       Source CloudXR env, ensure OpenXR runtime, then pass
                             --xr and --teleop_device handtracking to IsaacLab.
  --xr-view first|third      SONIC_XR_VIEW. Default: first.
  --cloudxr-install-dir PATH CloudXR install directory. Default: ~/.cloudxr.
  --cloudxr-env PATH         CloudXR env file sourced before IsaacLab when --xr is set.
                             Default: <cloudxr-install-dir>/run/cloudxr.env.
  --cloudxr-python PATH      Python used to auto-start CloudXR runtime.
                             Default: ~/miniconda3/bin/python, then python3/python.
  --cloudxr-timeout SEC      Seconds to wait for auto-started runtime. Default: 30.
  --cloudxr-setup-oob        Pass --setup-oob when auto-starting CloudXR runtime.
  --no-cloudxr-autostart     With --xr, only source/check env; do not start runtime.
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

expand_path() {
    local path="$1"
    case "${path}" in
        "~")
            printf '%s\n' "${HOME}"
            ;;
        "~/"*)
            printf '%s/%s\n' "${HOME}" "${path#~/}"
            ;;
        *)
            printf '%s\n' "${path}"
            ;;
    esac
}

default_cloudxr_python() {
    if [[ -x "${HOME}/miniconda3/bin/python" ]]; then
        printf '%s\n' "${HOME}/miniconda3/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    fi
}

validate_positive_int() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a positive integer"
    (( value > 0 )) || die "${name} must be a positive integer"
}

cloudxr_run_dir() {
    printf '%s\n' "${NV_CXR_RUNTIME_DIR:-${cloudxr_install_dir}/run}"
}

source_cloudxr_env() {
    [[ -f "${cloudxr_env}" ]] || return 1
    set -a
    # shellcheck disable=SC1090
    . "${cloudxr_env}"
    set +a
}

cloudxr_runtime_ready() {
    local run_dir
    run_dir="$(cloudxr_run_dir)"
    [[ -e "${run_dir}/runtime_started" ]] || return 1
    [[ -S "${run_dir}/ipc_cloudxr" ]] || return 1
}

wait_for_cloudxr_runtime() {
    local timeout="$1"
    local deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
        if cloudxr_runtime_ready; then
            return 0
        fi
        sleep 1
    done
    return 1
}

tail_cloudxr_log() {
    local log_path="$1"
    [[ -f "${log_path}" ]] || return 0
    log "CloudXR launcher log tail:"
    tail -n 40 "${log_path}" >&2 || true
}

start_cloudxr_runtime() {
    [[ -n "${cloudxr_python}" ]] || die "No python found for CloudXR; pass --cloudxr-python"
    [[ -x "${cloudxr_python}" ]] || die "CloudXR python is not executable: ${cloudxr_python}"

    local run_dir="${cloudxr_install_dir}/run"
    local logs_dir="${cloudxr_install_dir}/logs"
    mkdir -p "${run_dir}" "${logs_dir}"

    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    local log_path="${logs_dir}/isaacteleop_cloudxr_${timestamp}.log"
    local -a cloudxr_cmd=(
        "${cloudxr_python}"
        "-m"
        "isaacteleop.cloudxr"
        "--cloudxr-install-dir"
        "${cloudxr_install_dir}"
        "--accept-eula"
    )
    if (( cloudxr_setup_oob )); then
        cloudxr_cmd+=("--setup-oob")
    fi

    log "CloudXR runtime is not ready; starting it in background"
    print_command "${cloudxr_cmd[@]}"
    nohup "${cloudxr_cmd[@]}" >"${log_path}" 2>&1 &
    local launcher_pid=$!
    printf '%s\n' "${launcher_pid}" >"${run_dir}/isaacteleop_cloudxr_launcher.pid" || true

    if ! wait_for_cloudxr_runtime "${cloudxr_timeout}"; then
        tail_cloudxr_log "${log_path}"
        die "CloudXR runtime did not become ready within ${cloudxr_timeout}s"
    fi
    log "CloudXR runtime started; launcher pid: ${launcher_pid}; log: ${log_path}"
}

prepare_cloudxr_for_xr() {
    cloudxr_install_dir="$(expand_path "${cloudxr_install_dir}")"
    if [[ -z "${cloudxr_env}" ]]; then
        cloudxr_env="${cloudxr_install_dir}/run/cloudxr.env"
    else
        cloudxr_env="$(expand_path "${cloudxr_env}")"
    fi
    cloudxr_python="$(expand_path "${cloudxr_python}")"

    export TELEOP_PROXY_HOST="${TELEOP_PROXY_HOST:-${local_ip}}"
    export TELEOP_STREAM_SERVER_IP="${TELEOP_STREAM_SERVER_IP:-${local_ip}}"

    if [[ -f "${cloudxr_env}" ]]; then
        source_cloudxr_env
    elif (( ! dry_run && ! cloudxr_autostart )); then
        die "CloudXR env file not found: ${cloudxr_env}"
    fi

    if cloudxr_runtime_ready; then
        log "CloudXR runtime: ready"
    elif (( dry_run )); then
        if (( cloudxr_autostart )); then
            log "CloudXR runtime: not ready (dry-run; would auto-start before IsaacLab)"
        else
            log "CloudXR runtime: not ready (dry-run; autostart disabled)"
        fi
    elif (( cloudxr_autostart )); then
        start_cloudxr_runtime
        source_cloudxr_env || die "CloudXR env file not found after runtime start: ${cloudxr_env}"
    else
        die "CloudXR runtime is not ready; start it first or remove --no-cloudxr-autostart"
    fi

    if [[ -z "${XR_RUNTIME_JSON:-}" ]]; then
        if (( dry_run )); then
            log "XR_RUNTIME_JSON: <unset> (dry-run; expected from ${cloudxr_env})"
        else
            die "XR_RUNTIME_JSON is unset; source ${cloudxr_env} before IsaacLab"
        fi
    elif [[ ! -f "${XR_RUNTIME_JSON}" ]]; then
        die "XR_RUNTIME_JSON points to a missing file: ${XR_RUNTIME_JSON}"
    else
        log "XR_RUNTIME_JSON: ${XR_RUNTIME_JSON}"
    fi

    log "CloudXR env: ${cloudxr_env}"
    log "TELEOP_PROXY_HOST: ${TELEOP_PROXY_HOST}, TELEOP_STREAM_SERVER_IP: ${TELEOP_STREAM_SERVER_IP}"
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
auto_recover="1"
target_rate_limit="0.04"
xr_view="first"
cloudxr_install_dir="${HOME}/.cloudxr"
cloudxr_env=""
cloudxr_python="$(default_cloudxr_python)"
cloudxr_timeout="30"

headless=0
xr=0
enable_pinocchio=0
dry_run=0
cloudxr_autostart=1
cloudxr_setup_oob=0

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
        --auto-recover)
            need_value "$1" "${2-}"
            auto_recover="$2"
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
        --cloudxr-install-dir)
            need_value "$1" "${2-}"
            cloudxr_install_dir="$2"
            shift 2
            ;;
        --cloudxr-env)
            need_value "$1" "${2-}"
            cloudxr_env="$2"
            shift 2
            ;;
        --cloudxr-python)
            need_value "$1" "${2-}"
            cloudxr_python="$2"
            shift 2
            ;;
        --cloudxr-timeout)
            need_value "$1" "${2-}"
            cloudxr_timeout="$2"
            shift 2
            ;;
        --cloudxr-setup-oob)
            cloudxr_setup_oob=1
            shift
            ;;
        --no-cloudxr-autostart)
            cloudxr_autostart=0
            shift
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
validate_01 "--auto-recover" "${auto_recover}"
validate_positive_int "--cloudxr-timeout" "${cloudxr_timeout}"

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

if (( xr )); then
    prepare_cloudxr_for_xr
fi

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
export SONIC_DEPLOY_AUTO_RECOVER="${auto_recover}"
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
    # Construct OpenXRDevice so dynamic anchors and optional XRCore button events
    # are active. "handtracking" is the env_cfg.teleop_devices key; startup yaw
    # recenter and optional B-button recenter do not require motion-controller
    # retargeters.
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
