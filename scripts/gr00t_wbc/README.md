# GR00T WBC closed-loop Isaac Lab simulation

This runner makes Isaac Lab/PhysX the robot state source and plant. The GR00T
deployment keeps its production Unitree DDS `LowState -> policy -> LowCmd` path.

Data flow:

```text
Isaac Lab PhysX state --ZMQ--> isaaclab_dds_bridge --DDS LowState--> GR00T WBC
Isaac Lab actuators   <--ZMQ-- isaaclab_dds_bridge <--DDS LowCmd--- GR00T WBC
```

The runner holds the reset root pose for at least 3.5 seconds and continues to
hold it while the deployment reports `INIT` or `WAIT_FOR_CONTROL`. It releases
the root only after the deployment reports `CONTROL`. During normal WBC control,
gravity, contacts, locomotion, and falls are PhysX results. When a fall is detected,
the runner resets G1, temporarily supports the root at the standing pose, and then
smoothly transfers control back to WBC.

## Build the bridge and deployment

```bash
cd /home/nolovr/GR00T-WholeBodyControl/gear_sonic_deploy
cmake -S . -B build
cmake --build build --target isaaclab_dds_bridge g1_deploy_onnx_ref -j2
```

## Start three terminals

Terminal 1, DDS/ZMQ bridge:

```bash
cd /home/nolovr/GR00T-WholeBodyControl/gear_sonic_deploy
./target/release/isaaclab_dds_bridge lo 5560 5561
```

Terminal 2, WBC with a bundled reference-motion input:

```bash
cd /home/nolovr/GR00T-WholeBodyControl/gear_sonic_deploy
./target/release/g1_deploy_onnx_ref \
  lo \
  policy/release/model_decoder.onnx \
  reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --disable-crc-check \
  --auto-start \
  --input-type keyboard \
  --zmq-out-port 5557
```

For skeleton/VR streaming, keep `--auto-start` and replace the input options
with the GR00T input endpoint, for example `--input-type zmq` or
`--input-type zmq_manager` plus `--zmq-host`, `--zmq-port`, and `--zmq-topic`.

Terminal 3, Isaac Lab with a window:

```bash
cd /home/nolovr/IsaacLab
TERM=xterm-256color ./isaaclab.sh -p scripts/gr00t_wbc/run_g1_wbc_closed_loop.py \
  --device cuda:0
```

For a headless smoke test, append `--headless`.

## Safety behavior

- No WBC action before the startup deadline: exit.
- Action stream older than 200 ms by default: exit.
- Root below the configured minimum height: reset to the standing pose, hold for
  2 seconds by default, then blend back to WBC. Pass `--exit-on-fall` to restore
  the fail-fast behavior for debugging.
- Non-finite or non-29-DoF action: reject and exit.
- ZMQ state older than 200 ms: the bridge stops publishing DDS LowState, so
  the deployment's existing LowState safety timeout stops control.
