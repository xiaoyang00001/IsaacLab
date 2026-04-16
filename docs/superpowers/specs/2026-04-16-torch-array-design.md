# TorchArray — Warp-First Dual-Access Array

**Date**: 2026-04-16
**Author**: Antoine Richard
**Branch**: `antoiner/feat/TorchArray`
**Scope**: Articulation data (PhysX + Newton). Follow-up PRs extend to other assets/sensors.
**Attribution**: Inspired by `TorchArray` from [mujocolab/mjlab](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sim/sim_data.py) (BSD-3-Clause).

## Problem

All IsaacLab asset and sensor data properties return raw `wp.array`. Users must manually
call `wp.to_torch()` at every access point — over 551 occurrences in `isaaclab_tasks` alone.
This is verbose, error-prone, and obscures user code.

At the same time, the project is moving toward more warp-centric and CUDA-graph-friendly
workflows. A torch-first wrapper (like mjlab's original `TorchArray`) would conflict with
that direction: an object that silently behaves as `torch.Tensor` but fails when passed to
`wp.launch()` is ambiguous and misleading.

## Design

A `TorchArray` class that wraps a `wp.array` and provides **two explicit, unambiguous
accessors**:

```python
data.joint_pos          # TorchArray (the bridge object)
data.joint_pos.torch    # torch.Tensor — cached, zero-copy view
data.joint_pos.warp     # wp.array — the underlying buffer
```

There is no implicit behavior in the final state. Users choose which world they are in at
the point of access.

### Deprecation Bridge

To ease migration from the current `wp.to_torch(data.X)` pattern, the initial release
includes `__torch_function__` and arithmetic/comparison operators that work but emit a
one-time deprecation warning directing users to `.torch`:

```python
# All three work during the deprecation period:
pos = data.joint_pos.torch              # preferred — no warning
pos = data.joint_pos + offset           # works, warns: "use .torch"
pos = torch.cat([data.joint_pos], -1)   # works, warns: "use .torch"
```

In a future release, the implicit conversions are removed. Only `.torch` and `.warp` remain.

## TorchArray Class

**Location**: `source/isaaclab/isaaclab/utils/warp/torch_array.py`

```python
class TorchArray:
    """Warp array with explicit torch interoperability.

    Provides two explicit accessors:

    - ``.torch`` — cached zero-copy ``torch.Tensor`` view
    - ``.warp`` — the underlying ``wp.array``

    During the deprecation period, implicit torch operations (arithmetic,
    ``__torch_function__``) work but emit a one-time warning directing users
    to ``.torch``.
    """

    _deprecation_warned: ClassVar[bool] = False

    def __init__(self, wp_array: wp.array) -> None:
        self._warp = wp_array
        self._torch_cache: torch.Tensor | None = None

    @property
    def torch(self) -> torch.Tensor:
        """Cached zero-copy torch.Tensor view."""
        if self._torch_cache is None:
            self._torch_cache = wp.to_torch(self._warp)
        return self._torch_cache

    @property
    def warp(self) -> wp.array:
        """The underlying warp array."""
        return self._warp

    # Convenience pass-throughs (no deprecation — these are informational)
    @property
    def shape(self) -> tuple: ...
    @property
    def dtype(self): ...
    @property
    def device(self): ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

    # --- Deprecation bridge (to be removed in a future release) ---

    @classmethod
    def _warn_implicit(cls) -> None:
        """Emit a one-time deprecation warning for implicit torch conversion."""
        ...

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch.* calls, unwrap to .torch, warn."""
        ...

    # Arithmetic operators: __add__, __radd__, __sub__, __rsub__,
    # __mul__, __rmul__, __truediv__, __rtruediv__, __pow__, __rpow__,
    # __neg__, __pos__, __abs__
    #
    # Comparison operators: __eq__, __ne__, __lt__, __le__, __gt__, __ge__
    #
    # All delegate to self.torch with deprecation warning.
```

### Convenience Properties

`shape`, `dtype`, and `device` are exposed directly (no deprecation) since they are
informational and unambiguous — they describe the warp array:

- `shape` — from the warp array
- `dtype` — warp dtype
- `device` — warp device string

`__repr__` shows both sides: `TorchArray(shape=..., warp_dtype=..., device=...)`.

`__len__` returns the first dimension size (from the warp array).

### What `shape` Means

The warp array may use structured types (e.g., `wp.vec3f`, `wp.quatf`). The `shape`
property returns the **warp shape**, which does not include the trailing component
dimension. For example, a `(N,) wp.vec3f` array has `shape = (N,)` but its `.torch`
view has shape `(N, 3)`.

This is intentional — the object is warp-first. Users who need torch shapes use
`.torch.shape`.

## Integration with Articulation Data

### Property Changes

Every `@property` that currently returns `wp.array` wraps the result in `TorchArray()`
before returning. The base ABC type hints change from `-> wp.array` to `-> TorchArray`.

**Buffered properties** (joint_pos, joint_vel, body_link_pose_w, etc.):

```python
# Before:
@property
def joint_pos(self) -> wp.array:
    if self._joint_pos.timestamp < self._sim_timestamp:
        self._joint_pos.data = self._root_view.get_dof_positions()
        self._joint_pos.timestamp = self._sim_timestamp
    return self._joint_pos.data

# After:
@property
def joint_pos(self) -> TorchArray:
    if self._joint_pos.timestamp < self._sim_timestamp:
        self._joint_pos.data = self._root_view.get_dof_positions()
        self._joint_pos.timestamp = self._sim_timestamp
    return TorchArray(self._joint_pos.data)
```

**Sliced properties** (root_link_pos_w, root_link_quat_w, etc.):

These call internal helpers like `_get_pos_from_transform` which expect `wp.array`.
They access `.warp` on the parent TorchArray:

```python
# Before:
@property
def root_link_pos_w(self) -> wp.array:
    return self._get_pos_from_transform(self.root_link_pose_w)

# After:
@property
def root_link_pos_w(self) -> TorchArray:
    return TorchArray(self._get_pos_from_transform(self.root_link_pose_w.warp))
```

**Direct-return properties** (default_root_pose, joint_pos_target, etc.):

```python
# Before:
@property
def default_root_pose(self) -> wp.array:
    return self._default_root_pose

# After:
@property
def default_root_pose(self) -> TorchArray:
    return TorchArray(self._default_root_pose)
```

### What Does NOT Change

- **Internal warp kernels**: All `wp.launch()` calls use raw `wp.array` buffers
  (`self._buffer.data`, `self.GRAVITY_VEC_W`, etc.) — unchanged.
- **Write methods**: `set_joint_position_target()` etc. take `torch.Tensor` args and do
  `wp.from_torch()` internally — unchanged.
- **Setters**: `default_root_pose.setter` accepts `wp.array` — unchanged.
- **`_create_buffers()`**: All `TimestampedBufferWarp` usage — unchanged.
- **Derived property kernels**: `projected_gravity_b`, `heading_w`, etc. call
  `wp.launch()` with internal buffers and access parent properties via `.warp`.

### Files Modified

| File | Change |
|------|--------|
| `isaaclab/utils/warp/torch_array.py` | **New** — `TorchArray` class |
| `isaaclab/utils/warp/__init__.py` | Export `TorchArray` |
| `isaaclab/assets/articulation/base_articulation_data.py` | Return types `wp.array` → `TorchArray` |
| `isaaclab_physx/assets/articulation/articulation_data.py` | Wrap returns in `TorchArray()`, derived props use `.warp` |
| `isaaclab_newton/assets/articulation/articulation_data.py` | Same as PhysX |
| `isaaclab/docs/CHANGELOG.rst` | Added entry |
| `isaaclab_physx/docs/CHANGELOG.rst` | Added entry |
| `isaaclab_newton/docs/CHANGELOG.rst` | Added entry |

### Setters and the TorchArray

Property setters (e.g., `default_root_pose.setter`) currently accept `wp.array`. These
remain unchanged — they accept `wp.array` and do `.assign()`. If a user passes a
`TorchArray`, the setter will fail. This is acceptable: setters are used during
initialization, not in hot loops, and the error is clear.

## Migration Guide

### For users doing `wp.to_torch(data.X)`

```python
# Before:
pos = wp.to_torch(robot.data.root_link_pos_w)

# After:
pos = robot.data.root_link_pos_w.torch
```

### For users passing data to warp kernels

```python
# Before:
wp.launch(kernel, inputs=[robot.data.joint_pos], ...)

# After:
wp.launch(kernel, inputs=[robot.data.joint_pos.warp], ...)
```

### For users doing arithmetic on data properties

```python
# Works during deprecation period (with warning):
scaled = robot.data.joint_pos * scale_factor

# Preferred:
scaled = robot.data.joint_pos.torch * scale_factor
```

## Testing Strategy

1. **Unit tests for `TorchArray`**:
   - `.torch` returns correct `torch.Tensor` with zero-copy (shared memory)
   - `.warp` returns the original `wp.array`
   - `.torch` is cached (same object on repeated access)
   - `__torch_function__` works and emits deprecation warning
   - Arithmetic operators work and emit deprecation warning
   - Comparison operators work and emit deprecation warning
   - `shape`, `dtype`, `device`, `len` return warp-side values
   - Works with structured types (`wp.vec3f`, `wp.quatf`, `wp.transformf`, `wp.spatial_vectorf`)

2. **Articulation integration tests**:
   - All existing articulation tests pass with minimal changes
     (`wp.to_torch(data.X)` → `data.X.torch`)
   - Verify `.warp` can be passed to `wp.launch()` successfully

3. **Validation — run existing test suites**:
   - PhysX mock articulation view: `./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view.py`
   - PhysX mock articulation view (warp): `./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view_warp.py`
   - Newton mock articulation view: `./isaaclab.sh -p -m pytest source/isaaclab_newton/test/test_mock_interfaces/test_mock_articulation_view.py`
   - PhysX articulation asset tests (requires GPU sim): `./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py`
   - Newton articulation asset tests (requires GPU sim): `./isaaclab.sh -p -m pytest source/isaaclab_newton/test/assets/test_articulation.py`
   - Pre-commit checks: `./isaaclab.sh -f`

## Future Work

- Extend to `RigidObject`, `RigidObjectCollection`, `DeformableObject`
- Extend to sensors (`ContactSensor`, `Imu`, `Pva`, `FrameTransformer`)
- Remove deprecation bridge in a subsequent release
