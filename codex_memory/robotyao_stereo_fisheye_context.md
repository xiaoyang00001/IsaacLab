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

## Notes

- A previous edit was made to the IsaacLab integration template copy under `D:\Omniverse\IsaacLab\integrations\robotyao_unity`, but the user clarified the actual running Unity project is `D:\UnityWork\RobotYao`. The real Unity project must be changed for the fix to affect the running scene.
- Do not assume edits under `D:\Omniverse\IsaacLab\integrations\robotyao_unity` affect the Unity Editor session.
- Before the Unity fix, `SampleScene.unity` contained a serialized `flipY: 1`; this has been changed to `flipY: 0`.
