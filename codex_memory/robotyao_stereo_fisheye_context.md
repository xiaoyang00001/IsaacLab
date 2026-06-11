# RobotYao Stereo Fisheye Context

Last updated: 2026-06-09

## Project Paths

- IsaacLab workspace: `D:\Omniverse\IsaacLab`
- IsaacLab streaming script: `D:\Omniverse\IsaacLab\scripts\robotyao\stream_stereo_fisheye_zmq.py`
- Unity project: `D:\UnityWork\RobotYao`
- Unity receiver script: `D:\UnityWork\RobotYao\Assets\Scripts\RobotYao\RobotYaoStereoFisheyeApp.cs`
- Unity panorama player: `D:\UnityWork\RobotYao\Assets\Scripts\RobotYao\RobotYaoStereoPanoramaPlayer.cs`
- Unity fisheye shader: `D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180Inside.shader`
- Unity stereo shaders:
  - `D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180StereoInside.shader`
  - `D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180StereoSkybox.shader`
- C++ native plugin project: `E:\VSCode\RobotoYaoS`

## Main Run Command

```powershell
.\isaaclab.bat -p .\scripts\robotyao\stream_stereo_fisheye_zmq.py --task-scene --task-use-rmpflow --unity-control --width 1920 --height 1920 --fps 30 --encoding h264 --show-camera-lenses
```

## Desired Camera Mount

The stereo fisheye rig must be strictly fixed to the robot head model, under:

```text
/World/envs/env_*/Robot/link_pitch_head/RobotYaoTaskStereo
```

The user-provided transform for `RobotYaoTaskStereo` is:

```text
Translate: X=-0.25597, Y=0.15846, Z=0.0
Orient:    X=-90.0,    Y=0.0,     Z=180.0
Scale:     X=1.0,    Y=1.0,     Z=1.0
```

With the current `RobotYaoTaskStereo` transform, visual verification showed positive local Y appears on the robot's physical right side. Head-link camera offsets use:

```text
LeftFisheye local Y  = -baseline / 2
RightFisheye local Y = +baseline / 2
```

The visual lens balls must be centered exactly at the actual camera origins. The previous structure with `Left` / `Right` Xforms and `Fisheye` children caused confusion because the visible balls were not centered on the actual camera prim origin.

Preferred head-link structure:

```text
/World/envs/env_*/Robot/link_pitch_head/RobotYaoTaskStereo
  /LeftFisheye        Camera
    /LensVisual       visual sphere
  /RightFisheye       Camera
    /LensVisual       visual sphere
```

## IsaacLab Changes Already Made

In `scripts\robotyao\stream_stereo_fisheye_zmq.py`:

- Task camera mount default was changed to head-link mode.
- Head rig default transform was changed to:
  - `--task-camera-head-rig-x -0.25597`
  - `--task-camera-head-rig-y 0.15846`
  - `--task-camera-head-rig-z 0.0`
  - `--task-camera-head-rig-roll-deg -90.0`
  - `--task-camera-head-rig-pitch-deg 0.0`
  - `--task-camera-head-rig-yaw-deg 180.0`
- Head-link mode was refactored to create `LeftFisheye` and `RightFisheye` camera prims directly under `RobotYaoTaskStereo`.
- Head-link left/right camera local Y offsets are currently: `LeftFisheye=-baseline/2`, `RightFisheye=+baseline/2`.
- `LensVisual` is now a child of each Camera prim so the colored visual marker center matches the fisheye camera origin.
- Added helper logic for tuple cameras so the task loop can initialize, reset, and stream two independent `Camera` objects.
- Verified Python syntax with:

```powershell
python -m py_compile .\scripts\robotyao\stream_stereo_fisheye_zmq.py
```

## Unity Upside-Down Panorama Fix

Unity showed the received fisheye panorama upside down: sky/ground were vertically swapped. IsaacLab viewport appeared normal.

Important finding:

- The real C++ plugin at `E:\VSCode\RobotoYaoS` uses FFmpeg `sws_scale` with `dstStride = expectedWidth * 4`, then D3D11 `UpdateSubresource(..., rowPitch = width * 4)`. It does not flip rows.
- Unity shader `RobotYao/Fisheye180Inside` flips the sampled UV when `_FlipY > 0.5`.
- Unity C# script had serialized `flipY = true` in the real project, and the open scene also serialized `flipY: 1`.

Applied fix in the real Unity project:

- Set `RobotYaoStereoFisheyeApp.flipY` default to `false`.
- Set `_FlipY` default to `0` in relevant fisheye shaders.
- Updated `Assets\Scenes\SampleScene.unity` serialized `flipY: 1` to `flipY: 0`, otherwise the scene instance overrides the script default.

Changed files in `D:\UnityWork\RobotYao`:

```text
Assets\Scripts\RobotYao\RobotYaoStereoFisheyeApp.cs
Assets\Shaders\RobotYao\Fisheye180Inside.shader
Assets\Shaders\RobotYao\Fisheye180StereoInside.shader
Assets\Shaders\RobotYao\Fisheye180StereoSkybox.shader
Assets\Scenes\SampleScene.unity
```

## Unity Stereo Eye Order And Color Fix

Unity later showed a cross-eyed stereo effect and visibly wrong VR180 reprojection, especially on near objects. A first attempt used shader `_SwapEyes`, then hard texture-binding swap, but the visual result was still wrong. The current correction uses horizontal fisheye UV flip plus hard texture-binding eye swap:

```text
D:\UnityWork\RobotYao\Assets\Scripts\RobotYao\RobotYaoStereoFisheyeApp.cs
  flipX = true
  swapRedBlue = true
  swapStereoEyes = true

D:\UnityWork\RobotYao\Assets\Scenes\SampleScene.unity
  flipY: 0
  flipX: 1
  swapRedBlue: 1
  swapStereoEyes: 1

D:\UnityWork\RobotYao\Assets\Scripts\RobotYao\RobotYaoStereoPanoramaPlayer.cs
  _FlipX = 1 when settings.flipX is enabled
  _SwapRedBlue = 1 when settings.swapRedBlue is enabled
  _FovDeg, _FlipX, _FlipY, _SwapRedBlue, yaw/pitch, and hard stereo texture bindings are refreshed every frame
  _SwapEyes remains 0 because eye swapping is done by binding _LeftTex/_RightTex directly

D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180StereoInside.shader
  _FlipX default = 1
  _SwapRedBlue default = 1
  _SwapEyes default = 0

D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180StereoSkybox.shader
  _FlipX default = 1
  _SwapRedBlue default = 1
  _SwapEyes default = 0

D:\UnityWork\RobotYao\Assets\Shaders\RobotYao\Fisheye180Inside.shader
  _FlipX default = 1
  _SwapRedBlue default = 1
```

Color mismatch was diagnosed from the screenshot where the blue Isaac box appeared orange/peach in Unity. The native plugin decodes H264/JPEG to BGRA and uploads that texture memory to Unity, while the shader path samples as RGBA. `swapRedBlue=true` makes the shader apply `color = color.bgra`, which should restore the blue box color.

`RobotYaoStereoFisheyeApp` now stores `RobotYaoStereoPanoramaSettings` as a member field and calls `SyncPanoramaSettings()` from `LateUpdate()`. This is important because Inspector changes during Play Mode previously changed the MonoBehaviour fields but did not update the already-created panorama player settings object.

Do not fix this by moving the Isaac camera positions unless the lens visuals are physically wrong again. The Isaac stream still publishes `left_payload, right_payload`; Unity mirrors the fisheye UV horizontally (`uv.x = 1 - uv.x`) before sampling and currently swaps eyes by binding `_LeftTex` to the right source and `_RightTex` to the left source when `swapStereoEyes=true`.

Verified on 2026-06-09:

```powershell
dotnet build D:\UnityWork\RobotYao\RobotYao.sln --no-restore
```

The build completed with `0` errors and existing Unity/AVPro warnings.

## Isaac Fisheye Projection Alignment

After the color fix, Unity colors matched Isaac, but the headset view still showed a strong cross-eyed stereo effect and possible projection deformation. The likely cause was not RGB channel order or camera placement, but a mismatch between Isaac's fisheye lens model and Unity's shader reprojection model.

Unity shaders currently assume an equidistant 180-degree circular fisheye:

```text
radius = theta / maxTheta
uv = center + radialDir * radius * 0.5
```

Isaac was using `projection_type="fisheyePolynomial"` without explicit polynomial coefficients. IsaacLab defaults include `fisheye_polynomial_b = 0.00245`, which does not match a 1920x1920 full-frame 180-degree circle. For a full-frame f-theta/equidistant image:

```text
poly_b = radians(fisheye_fov / 2) / (min(width, height) / 2)
```

For the user's command (`--width 1920 --height 1920 --fisheye_fov 180`) this is:

```text
poly_b = 0.001636246
```

Applied in `scripts\robotyao\stream_stereo_fisheye_zmq.py`:

- Added `_fisheye_full_frame_poly_b(width, height, fisheye_fov)`.
- Both built-in and task-scene fisheye cameras now explicitly set:
  - `fisheye_polynomial_a = 0.0`
  - `fisheye_polynomial_b = _fisheye_full_frame_poly_b(...)`
  - `fisheye_polynomial_c = 0.0`
  - `fisheye_polynomial_d = 0.0`
  - `fisheye_polynomial_e = 0.0`
  - `fisheye_polynomial_f = 0.0`
- ZMQ header now includes `radius_px` and `poly_a..poly_f`.
- Task stereo mount startup log now prints `fisheye_poly_b`.

Verification:

```powershell
python -m py_compile .\scripts\robotyao\stream_stereo_fisheye_zmq.py
git diff --check -- .\scripts\robotyao\stream_stereo_fisheye_zmq.py
python -c "import math; print(f'{math.radians(180*0.5)/(1920*0.5):.9f}')"
```

The last command prints `0.001636246`.

## Current Stereo Comfort Analysis

After color correction and f-theta polynomial alignment, the user still reports a cross-eyed feeling. Latest visual evidence suggests the remaining issue may be excessive stereo disparity for close objects rather than RGB channel order. The grippers, cube, and tabletop are very close to the head-mounted fisheye origins, and the default Isaac baseline is still:

```text
--baseline 0.064
```

For close teleoperation content, a 64 mm stereo baseline can produce very large disparity, especially below roughly 0.5 m. Recommended next test is to run the same command with a smaller baseline first:

```powershell
--baseline 0.032
```

If still uncomfortable, test `--baseline 0.024` or `--baseline 0.016`. This keeps the cameras centered on the same head-mounted rig but moves the left/right lens centers closer together.

Also note that the current real Unity scene file was observed with:

```text
swapStereoEyes: 0
```

So left/right sign and baseline should be tested separately:

1. Toggle `Swap Stereo Eyes` live in Unity Play Mode to determine stereo sign.
2. If both signs still feel uncomfortable, reduce Isaac `--baseline`.

## Notes

- A previous edit was made to the IsaacLab integration template copy under `D:\Omniverse\IsaacLab\integrations\robotyao_unity`, but the user clarified the actual running Unity project is `D:\UnityWork\RobotYao`. The real Unity project must be changed for the fix to affect the running scene.
- Do not assume edits under `D:\Omniverse\IsaacLab\integrations\robotyao_unity` affect the Unity Editor session.
- Before the Unity fix, `SampleScene.unity` contained a serialized `flipY: 1`; this has been changed to `flipY: 0`.

## XR B Button Delta Follow Check

Current user request: check the feature where pressing the right-hand B button lets the robot arm follow XR controller movement in delta mode, especially Unity-to-Isaac coordinate conversion.

Input path verified:

```text
Unity right secondaryButton (B)
-> RobotYaoXrInputPublisher ButtonSecondary = bit1
-> C++ RobotYaoXrPublisher MGXR controller packet
-> ZeroMqGameSubDevice button_1_mask = 1 << 1
-> DeviceBase.MotionControllerInputIndex.BUTTON_1
-> RobotYaoWheeledXrRetargeter follow_start_button
```

Right-hand A is `BUTTON_0` and stops follow in toggle mode.

Latest coordinate ownership:

- Unity now sends controller data using the right-handed MGXR/OpenXR convention. The real project at `D:\UnityWork\RobotYao` currently writes `pose.positionZ = -position.z`, `pose.rotationZ = -rotation.z`, and `pose.rotationW = -rotation.w`.
- `ZeroMqGameSubDevice._pose_xyzw_to_wxyz()` is now the only OpenXR/MGXR-to-Isaac coordinate conversion point.
- `RobotYaoWheeledXrRetargeter` no longer applies a Unity-to-Isaac axis map. It computes `current_position - previous_position` directly in the already-converted Isaac xyz frame, then applies `--arm-delta-scale`.
- `--arm-unity-axis-map` has been removed from `scripts\robotyao\stream_stereo_fisheye_zmq.py` to avoid double conversion.

The ZMQ pose conversion is:

```text
MGXR/OpenXR: +X right, +Y up, -Z forward
Isaac Lab:   +X right, +Y forward, +Z up
Transform:   +90 degrees around X, applied in _pose_xyzw_to_wxyz()
```

Fix applied on 2026-06-09:

- `source\isaaclab\isaaclab\devices\openxr\retargeters\robotyao_wheeled_xr_retargeter.py`
  - Left/right controller deltas now update independently while arm-follow is active.
  - Previously, both left and right controller positions had to be valid or both deltas were cleared. That could make right-hand B follow fail if the left controller was not tracked.
- `scripts\robotyao\stream_stereo_fisheye_zmq.py`
  - `--debug-task-loop` now prints `follow`, `left_delta`, and `right_delta` every 10 task steps.

Verification commands run:

```powershell
python -m py_compile source\isaaclab\isaaclab\devices\openxr\retargeters\robotyao_wheeled_xr_retargeter.py scripts\robotyao\stream_stereo_fisheye_zmq.py source\isaaclab\isaaclab\devices\openxr\zeromq_game_sub_device.py
git diff --check -- source\isaaclab\isaaclab\devices\openxr\retargeters\robotyao_wheeled_xr_retargeter.py scripts\robotyao\stream_stereo_fisheye_zmq.py
```

Expected live check:

Run Isaac with `--debug-task-loop`. After pressing right B:

```text
follow=True
right_delta=[...]
```

Follow-up on user feedback: the live experience felt like hand left/right movement was driving Isaac up/down. Code inspection showed:

- Unity XR position values are meter-based.
- `RobotYaoWheeledXrRetargeter` keeps `--arm-delta-scale=1.0` by default, so the scene-space delta remains meter-for-meter.
- Agibot `RMPFlowActionCfg.scale=1.0`.
- `apply_delta_pose()` in `source\isaaclab\isaaclab\utils\math.py` does `target_pos = source_pos + delta_pose[:, 0:3]`, so RMPFlow relative position actions are also in meters.

The likely mismatch is the final Agibot Lula/RMPFlow action frame axis order, not Unity units. A fix was added in `scripts\robotyao\stream_stereo_fisheye_zmq.py`:

- New CLI option:

```powershell
--arm-rmpflow-axis-map y,-x,z
```

- Default is now `y,-x,z`, meaning controller/scene Isaac delta `[right, forward, up]` is sent to RMPFlow/robot delta `[forward, left, up]`.
- This maps Controller Isaac `+X` right to RMPFlow `-Y` right, Controller Isaac `+Y` forward to RMPFlow `+X` forward, and Controller Isaac `+Z` up to RMPFlow `+Z` up.
- Signed variants are supported, for example `--arm-rmpflow-axis-map y,x,z`, if the lateral sign is still reversed.
- `--debug-task-loop` now prints:

```text
right_scene_delta=[...]
right_rmpflow_delta=[...]
axis_map=y,-x,z
actual_right_ee_delta_w=[...]
right_rmpflow_cum=[...]
actual_right_ee_cum_w=[...]
```

Use this to verify scale live:

```text
If right_rmpflow_delta magnitude is about 0.10, the commanded relative target is 10 cm.
If actual_right_ee_delta_w magnitude is much smaller/larger over steady movement, the error is RMPFlow tracking/limits/dynamics, not Unity unit conversion.
For a 10 cm manual test, press B to start follow, move the controller about 10 cm, and compare right_rmpflow_cum against actual_right_ee_cum_w.
```

Further axis audit:

- Unity `SampleScene.unity` has `leftControllerPoseOverride` and `rightControllerPoseOverride` unset, so `RobotYaoXrInputPublisher` publishes raw XR `devicePosition`, not a scene Transform override.
- Unity `XR Origin (XR Rig)` scene override is identity position/rotation/scale.
- C++ project `E:\VSCode\RobotoYaoS` only copies Unity pose floats into the MGXR packet; it does not change axes.
- Unity visual playback does apply view/display transforms: `flipX=true`, skybox VR180, and `recenterPanoramaOnStart` adjusts panorama yaw from the XR camera yaw. Therefore the "direction seen in the Unity headset image" can differ from the raw XR tracking-space axes.

New diagnostic/flexibility added:

- `RobotYaoWheeledXrRetargeter` now appends unscaled Isaac-frame controller position deltas and scaled controller orientation deltas to its command tensor:

```text
RAW_LEFT_DELTA_START = 20
RAW_RIGHT_DELTA_START = 23
LEFT_ARM_ROT_DELTA_START = 26
RIGHT_ARM_ROT_DELTA_START = 29
OUTPUT_SIZE = 32
```

- Orientation is computed as a relative quaternion delta (`current * previous^-1`) and converted to axis-angle radians.
- New CLI option:

```powershell
--arm-rotation-delta-scale 1.0
```

- In the task-scene RMPFlow path, `actions[:, 0:3]` receives mapped position delta and `actions[:, 3:6]` receives mapped axis-angle rotation delta.
- The same `--arm-rmpflow-axis-map y,-x,z` is used for both position vectors and rotation vectors.

- `--debug-task-loop` now also prints:

```text
right_controller_delta_isaac=[...]
right_scene_delta=[...]
right_rmpflow_delta=[...]
right_scene_rot_delta=[...]
right_rmpflow_rot_delta=[...]
actual_right_ee_delta_w=[...]
right_rmpflow_cum=[...]
right_rmpflow_rot_cum=[...]
actual_right_ee_cum_w=[...]
```

Recommended live calibration:

1. Press B to enter follow mode.
2. Move the right controller only right/left, then only up/down, then only forward/back.
3. Rotate the right controller around one local/visible axis at a time and check `right_scene_rot_delta` and `right_rmpflow_rot_delta`.
4. Check which component changes in `right_controller_delta_isaac`, then which component changes in `right_scene_delta`, then which world component changes in `actual_right_ee_delta_w`.
5. If `right_controller_delta_isaac` is wrong, inspect Unity handedness conversion and `ZeroMqGameSubDevice._pose_xyzw_to_wxyz()`.
6. If `right_controller_delta_isaac`/`right_scene_rot_delta` are correct but `right_rmpflow_delta`/`right_rmpflow_rot_delta` or `actual_right_ee_delta_w` is wrong, change `--arm-rmpflow-axis-map`.

### Robot torso/body height control
A new vertical sliding height control feature has been implemented:
- **Condition**: Left-hand controller grip button (`left_inputs[SQUEEZE]`) is pressed (value > 0.5).
- **Control**: Left-hand thumbstick Y-axis (`THUMBSTICK_Y`) controls the sliding velocity of the robot torso along the wheeled base's slide rail (`joint_lift_body` joint target), overriding the default behavior of controlling forward/backward base translation. Meanwhile, the base forward and lateral translations are completely disabled and set to `0.0`.
- **Safety Constraints**: The slide rail joint target (`self._lift_joint_pos`) is dynamically updated and clamped to the robot's soft joint limits `self._robot.data.soft_joint_pos_limits[:, self._lift_joint_id]` retrieved at runtime to prevent exceeding physical rail limits.
- **Base Stability**: The wheeled base root Z coordinate (`self._root_pos[2]`) is kept strictly at its initial spawned height (`self._init_root_z`) to prevent gravity/penetration drift.
- **Default Behavior**: When the left grip button is not pressed, the left-hand thumbstick Y-axis and X-axis control the forward/backward and lateral translation of the wheeled robot base, preserving the original functionality.
- **Command Layout**:
  - `BASE_HEIGHT_VEL = 34` index added to the retargeter output.
  - `OUTPUT_SIZE` increased to `35`.

