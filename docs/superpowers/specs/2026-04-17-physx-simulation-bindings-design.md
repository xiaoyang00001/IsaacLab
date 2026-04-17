# PhysX `_create_simulation_bindings()` for ArticulationData

**Date**: 2026-04-17
**Author**: Antoine Richard
**Branch**: Stacked on `antoiner/feat/TorchArray`
**Scope**: PhysX ArticulationData only. Follow-up PRs extend to RigidObject/Collection.

## Problem

PhysX `ArticulationData` uses a lazy `TimestampedBufferWarp` pattern for direct
view-backed properties. Every access checks a timestamp, calls
`self._root_view.get_dof_positions()`, assigns to a buffer, and creates a new
`TorchArray`. We discovered the real PhysX view API returns the **same Python object**
with the **same pointer** on every call — the data updates in-place internally. The
timestamp check, buffer assignment, and TorchArray creation are all pure overhead for
these properties.

Newton already avoids this overhead with `_create_simulation_bindings()` — a pattern
that caches view getter results once and returns pinned TorchArrays directly. PhysX
should adopt the same pattern.

## Design

Introduce `_create_simulation_bindings()` in PhysX `ArticulationData`, mirroring
Newton's architecture:

1. At init, call each view getter once and store the result as a ``_sim_bind_*``
   attribute.
2. Pin a ``TorchArray`` on each binding.
3. Direct properties become one-liner returns — no timestamp, no refresh.
4. Computed/derived properties keep ``TimestampedBuffer`` + kernel launch but return
   pinned TorchArrays.
5. On reset, a ``PHYSICS_READY`` callback re-calls ``_create_simulation_bindings()``
   and rebinds all TorchArrays + invalidates sliced buffers.

## Property Categories

### Category A — Direct sim bindings (eliminate TimestampedBuffer)

These call a view getter that returns stable data updated in-place by PhysX. No
computation needed on the IsaacLab side.

```python
# Before (current):
@property
def joint_pos(self) -> TorchArray:
    if self._joint_pos.timestamp < self._sim_timestamp:
        self._joint_pos.data = self._root_view.get_dof_positions()
        self._joint_pos.timestamp = self._sim_timestamp
    return TorchArray(self._joint_pos.data)

# After:
@property
def joint_pos(self) -> TorchArray:
    return self._joint_pos_ta
```

Properties in this category:

- ``root_link_pose_w`` — from ``get_root_transforms().view(wp.transformf)``
- ``root_com_vel_w`` — from ``get_root_velocities().view(wp.spatial_vectorf)``
- ``body_link_pose_w`` — from ``get_link_transforms().view(wp.transformf)``
- ``body_com_vel_w`` — from ``get_link_velocities().view(wp.spatial_vectorf)``
- ``body_com_acc_w`` — from ``get_link_accelerations().view(wp.spatial_vectorf)``
- ``body_incoming_joint_wrench_b`` — from ``get_link_incoming_joint_force().view(wp.spatial_vectorf)``
- ``joint_pos`` — from ``get_dof_positions()``
- ``joint_vel`` — from ``get_dof_velocities()``

### Category A* — Direct sim bindings with extra step

- ``body_com_pose_b`` — from ``get_coms()`` which returns a **CPU** array. Requires
  ``wp.copy()`` to a pre-allocated GPU buffer each access. This property keeps a
  timestamp check to avoid redundant copies but returns a pinned TorchArray.

### Category B — Computed/derived (keep TimestampedBuffer, pin TorchArray)

These run warp kernels to derive values from category A inputs. The
``TimestampedBuffer`` pattern is kept for lazy evaluation, but the return changes
from ``TorchArray(self._buffer.data)`` to a pinned ``self._buffer_ta``.

- ``root_link_vel_w``, ``root_com_pose_w``, ``body_link_vel_w``, ``body_com_pose_w``
- ``projected_gravity_b``, ``heading_w``
- ``root_link_lin_vel_b``, ``root_link_ang_vel_b``
- ``root_com_lin_vel_b``, ``root_com_ang_vel_b``
- ``joint_acc``

### Category C — Pre-allocated constants (pin TorchArray)

These are allocated in ``_create_buffers()`` and never replaced. Just pin the
TorchArray.

- ``default_root_pose``, ``default_root_vel``, ``default_joint_pos``, ``default_joint_vel``
- ``joint_pos_target``, ``joint_vel_target``, ``joint_effort_target``
- ``computed_torque``, ``applied_torque``
- ``joint_stiffness``, ``joint_damping``, ``joint_armature``, ``joint_friction_coeff``
- ``joint_pos_limits``, ``joint_vel_limits``, ``joint_effort_limits``
- ``soft_joint_pos_limits``, ``soft_joint_vel_limits``
- ``gear_ratio``, ``body_mass``, ``body_inertia``

### Category D — Sliced (lazy pin, same as Newton)

These extract components from category A/B arrays via pointer arithmetic. Pin
lazily on first access.

- ``root_link_pos_w``, ``root_link_quat_w`` (from ``root_link_pose_w``)
- ``root_link_lin_vel_w``, ``root_link_ang_vel_w`` (from ``root_link_vel_w``)
- ``root_com_pos_w``, ``root_com_quat_w`` (from ``root_com_pose_w``)
- ``root_com_lin_vel_w``, ``root_com_ang_vel_w`` (from ``root_com_vel_w``)
- ``body_link_pos_w``, ``body_link_quat_w`` (from ``body_link_pose_w``)
- ``body_link_lin_vel_w``, ``body_link_ang_vel_w`` (from ``body_link_vel_w``)
- ``body_com_pos_w``, ``body_com_quat_w`` (from ``body_com_pose_w``)
- ``body_com_lin_vel_w``, ``body_com_ang_vel_w`` (from ``body_com_vel_w``)
- ``body_com_lin_acc_w``, ``body_com_ang_acc_w`` (from ``body_com_acc_w``)
- ``body_com_pos_b``, ``body_com_quat_b`` (from ``body_com_pose_b``)

## ``_create_simulation_bindings()``

New method on PhysX ``ArticulationData``:

```python
def _create_simulation_bindings(self) -> None:
    """Cache PhysX view getter results as direct simulation bindings.

    The PhysX view API returns the same wp.array object on every call —
    data is updated in-place by the solver. We cache these once and wrap
    them in TorchArrays for zero-overhead property access.
    """
    # Category A — direct view bindings
    self._sim_bind_root_link_pose_w = self._root_view.get_root_transforms().view(wp.transformf)
    self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities().view(wp.spatial_vectorf)
    self._sim_bind_body_link_pose_w = self._root_view.get_link_transforms().view(wp.transformf)
    self._sim_bind_body_com_vel_w = self._root_view.get_link_velocities().view(wp.spatial_vectorf)
    self._sim_bind_body_com_acc_w = self._root_view.get_link_accelerations().view(wp.spatial_vectorf)
    self._sim_bind_body_incoming_joint_wrench_b = (
        self._root_view.get_link_incoming_joint_force().view(wp.spatial_vectorf)
    )
    self._sim_bind_joint_pos = self._root_view.get_dof_positions()
    self._sim_bind_joint_vel = self._root_view.get_dof_velocities()

    # Rebind TorchArrays if they exist (skip first call from __init__)
    if hasattr(self, "_root_link_pose_w_ta"):
        self._root_link_pose_w_ta.rebind(self._sim_bind_root_link_pose_w)
        self._root_com_vel_w_ta.rebind(self._sim_bind_root_com_vel_w)
        # ... all category A TorchArrays
        # Invalidate category D sliced buffers (both _X_ta and _X)
        self._root_link_pos_w_ta = None
        self._root_link_pos_w = None
        # ... all sliced properties
```

## Rebind on Reset

Register ``PHYSICS_READY`` callback in ``Articulation._initialize_impl()``:

```python
self._physics_ready_handle = SimulationManager.register_callback(
    lambda _: self._data._create_simulation_bindings(),
    PhysicsEvent.PHYSICS_READY,
    name=f"articulation_rebind_{self.cfg.prim_path}",
)
```

The callback re-caches all view getters (which may return new objects after a full
reset) and rebinds TorchArrays. TimestampedBuffer timestamps are reset so computed
properties re-evaluate on next access.

## ``_create_buffers()`` Changes

- **Remove** ``TimestampedBufferWarp`` allocations for category A properties (they
  are replaced by ``_sim_bind_*`` + pinned TorchArrays).
- **Keep** ``TimestampedBufferWarp`` for category B (computed) properties.
- **Add** TorchArray pinning for all categories:
  - Category A: ``self._joint_pos_ta = TorchArray(self._sim_bind_joint_pos)``
  - Category B: ``self._root_link_vel_w_ta = TorchArray(self._root_link_vel_w.data)``
  - Category C: ``self._default_root_pose_ta = TorchArray(self._default_root_pose)``
  - Category D: ``self._root_link_pos_w_ta: TorchArray | None = None``

## ``update()`` Changes

The ``update(dt)`` method currently triggers lazy refresh by accessing
``self.joint_acc``. This still works since ``joint_acc`` is category B (computed)
and keeps the timestamp pattern. The ``_sim_timestamp`` increment is still needed for
category B properties.

**However**, category A properties no longer check timestamps. Their data is always
fresh because PhysX updates the underlying arrays in-place. So ``_sim_timestamp`` is
only relevant for category B and D.

## Files Modified

| File | Change |
|------|--------|
| ``isaaclab_physx/assets/articulation/articulation_data.py`` | Add ``_create_simulation_bindings()``, refactor ``_create_buffers()``, simplify category A properties, pin TorchArrays |
| ``isaaclab_physx/assets/articulation/articulation.py`` | Register ``PHYSICS_READY`` callback |

## Testing

All 210 existing PhysX articulation tests must pass **unchanged**. No new public API
surface — this is a pure internal optimization. The property signatures and return
types are identical.

## Validation Commands

```bash
# PhysX articulation tests (must all pass)
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -v --tb=short

# Mock view tests (should be unaffected)
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view_warp.py -v

# Pre-commit
./isaaclab.sh -f
```

## Future Work

- Extend to PhysX ``RigidObjectData`` and ``RigidObjectCollectionData``
- Extend to PhysX sensor data classes
- Investigate if ``body_com_pose_b`` (CPU → GPU copy) can be eliminated by requesting
  GPU-resident COMs from PhysX
