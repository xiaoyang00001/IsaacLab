# PhysX `_create_simulation_bindings()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-access overhead in PhysX ArticulationData by caching stable view getter results as direct simulation bindings, mirroring Newton's zero-overhead pattern.

**Architecture:** Add `_create_simulation_bindings()` to cache PhysX view getter results once (they return stable pointers). Pin TorchArrays on these bindings. Category A (direct) properties become one-liner returns. Category B (computed) properties keep TimestampedBuffer but return pinned TorchArrays. PHYSICS_READY callback handles rebinding after sim reset.

**Tech Stack:** Python, warp (`wp`), PhysX tensor API

**Spec:** `docs/superpowers/specs/2026-04-17-physx-simulation-bindings-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py` | Add `_create_simulation_bindings()`, refactor `_create_buffers()`, simplify properties, pin TorchArrays |
| Modify | `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py` | Register PHYSICS_READY callback |

---

### Task 1: Add `_create_simulation_bindings()` and refactor `_create_buffers()`

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`

- [ ] **Step 1: Add `_create_simulation_bindings()` method**

Add a new method after `__init__` that caches all PhysX view getter results. These are the Category A direct bindings:

```python
def _create_simulation_bindings(self) -> None:
    """Cache PhysX view getter results as direct simulation bindings.

    The PhysX view API returns the same wp.array object on every call —
    data is updated in-place by the solver. We cache these once and wrap
    them in TorchArrays for zero-overhead property access.
    """
    # Category A — direct view bindings (stable pointers, updated in-place by PhysX)
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

    # Rebind TorchArrays if they already exist (after sim reset)
    if hasattr(self, "_root_link_pose_w_ta"):
        # Category A rebinds
        self._root_link_pose_w_ta.rebind(self._sim_bind_root_link_pose_w)
        self._root_com_vel_w_ta.rebind(self._sim_bind_root_com_vel_w)
        self._body_link_pose_w_ta.rebind(self._sim_bind_body_link_pose_w)
        self._body_com_vel_w_ta.rebind(self._sim_bind_body_com_vel_w)
        self._body_com_acc_w_ta.rebind(self._sim_bind_body_com_acc_w)
        self._body_incoming_joint_wrench_b_ta.rebind(self._sim_bind_body_incoming_joint_wrench_b)
        self._joint_pos_ta.rebind(self._sim_bind_joint_pos)
        self._joint_vel_ta.rebind(self._sim_bind_joint_vel)
        # Reset TimestampedBuffer timestamps for Category B so kernels re-run
        self._root_link_vel_w.timestamp = -1.0
        self._root_com_pose_w.timestamp = -1.0
        self._body_link_vel_w.timestamp = -1.0
        self._body_com_pose_w.timestamp = -1.0
        self._body_com_pose_b.timestamp = -1.0
        self._projected_gravity_b.timestamp = -1.0
        self._heading_w.timestamp = -1.0
        self._root_link_lin_vel_b.timestamp = -1.0
        self._root_link_ang_vel_b.timestamp = -1.0
        self._root_com_lin_vel_b.timestamp = -1.0
        self._root_com_ang_vel_b.timestamp = -1.0
        self._joint_acc.timestamp = -1.0
        self._root_state_w.timestamp = -1.0
        self._root_link_state_w.timestamp = -1.0
        self._root_com_state_w.timestamp = -1.0
        self._body_state_w.timestamp = -1.0
        self._body_link_state_w.timestamp = -1.0
        self._body_com_state_w.timestamp = -1.0
        # Invalidate Category D sliced buffers (both TorchArray and backing wp.array)
        for attr in [
            "_root_link_pos_w", "_root_link_quat_w",
            "_root_link_lin_vel_w", "_root_link_ang_vel_w",
            "_root_com_pos_w", "_root_com_quat_w",
            "_root_com_lin_vel_w", "_root_com_ang_vel_w",
            "_body_link_pos_w", "_body_link_quat_w",
            "_body_link_lin_vel_w", "_body_link_ang_vel_w",
            "_body_com_pos_w", "_body_com_quat_w",
            "_body_com_lin_vel_w", "_body_com_ang_vel_w",
            "_body_com_lin_acc_w", "_body_com_ang_acc_w",
            "_body_com_pos_b", "_body_com_quat_b",
        ]:
            setattr(self, f"{attr}_ta", None)
            setattr(self, attr, None)
```

- [ ] **Step 2: Call `_create_simulation_bindings()` from `__init__`**

In the `__init__` method, add the call after `self._root_view` is set and before `self._create_buffers()`:

```python
def __init__(self, root_view, device: str):
    super().__init__(root_view, device)
    self._root_view = weakref.proxy(root_view)
    self._sim_timestamp = 0.0
    self._is_primed = False
    self._physics_sim_view = SimulationManager.get_physics_sim_view()
    # ... gravity/forward vec setup ...
    self._create_simulation_bindings()  # NEW: cache view getter results
    self._create_buffers()
```

- [ ] **Step 3: Refactor `_create_buffers()` — remove Category A TimestampedBuffers, add TorchArray pinning**

Remove the TimestampedBuffer allocations for Category A properties and replace with TorchArray pinning. Keep TimestampedBuffers for Category B.

Changes to `_create_buffers()`:

**Remove** these lines (Category A — now in `_create_simulation_bindings`):
```python
# DELETE:
self._root_link_pose_w = TimestampedBuffer(...)
self._root_com_vel_w = TimestampedBuffer(...)
self._body_link_pose_w = TimestampedBuffer(...)
self._body_com_vel_w = TimestampedBuffer(...)
self._body_com_acc_w = TimestampedBuffer(...)
self._body_incoming_joint_wrench_b = TimestampedBuffer(...)
self._joint_pos = TimestampedBuffer(...)
self._joint_vel = TimestampedBuffer(...)
```

**Keep** these (Category B — computed via kernels):
```python
# KEEP:
self._root_link_vel_w = TimestampedBuffer(...)
self._body_link_vel_w = TimestampedBuffer(...)
self._body_com_pose_b = TimestampedBuffer(...)
self._root_com_pose_w = TimestampedBuffer(...)
self._body_com_pose_w = TimestampedBuffer(...)
self._root_state_w = TimestampedBuffer(...)
# ... all other computed/state buffers
self._joint_acc = TimestampedBuffer(...)
self._projected_gravity_b = TimestampedBuffer(...)
self._heading_w = TimestampedBuffer(...)
self._root_link_lin_vel_b = TimestampedBuffer(...)
# etc.
```

**Add** TorchArray pinning at the end of `_create_buffers()`:
```python
# Pin TorchArrays — Category A (sim bindings)
self._root_link_pose_w_ta = TorchArray(self._sim_bind_root_link_pose_w)
self._root_com_vel_w_ta = TorchArray(self._sim_bind_root_com_vel_w)
self._body_link_pose_w_ta = TorchArray(self._sim_bind_body_link_pose_w)
self._body_com_vel_w_ta = TorchArray(self._sim_bind_body_com_vel_w)
self._body_com_acc_w_ta = TorchArray(self._sim_bind_body_com_acc_w)
self._body_incoming_joint_wrench_b_ta = TorchArray(self._sim_bind_body_incoming_joint_wrench_b)
self._joint_pos_ta = TorchArray(self._sim_bind_joint_pos)
self._joint_vel_ta = TorchArray(self._sim_bind_joint_vel)

# Pin TorchArrays — Category B (computed, wrap .data)
self._root_link_vel_w_ta = TorchArray(self._root_link_vel_w.data)
self._body_link_vel_w_ta = TorchArray(self._body_link_vel_w.data)
self._body_com_pose_b_ta = TorchArray(self._body_com_pose_b.data)
self._root_com_pose_w_ta = TorchArray(self._root_com_pose_w.data)
self._body_com_pose_w_ta = TorchArray(self._body_com_pose_w.data)
self._root_state_w_ta = TorchArray(self._root_state_w.data)
self._root_link_state_w_ta = TorchArray(self._root_link_state_w.data)
self._root_com_state_w_ta = TorchArray(self._root_com_state_w.data)
self._body_state_w_ta = TorchArray(self._body_state_w.data)
self._body_link_state_w_ta = TorchArray(self._body_link_state_w.data)
self._body_com_state_w_ta = TorchArray(self._body_com_state_w.data)
self._joint_acc_ta = TorchArray(self._joint_acc.data)
self._projected_gravity_b_ta = TorchArray(self._projected_gravity_b.data)
self._heading_w_ta = TorchArray(self._heading_w.data)
self._root_link_lin_vel_b_ta = TorchArray(self._root_link_lin_vel_b.data)
self._root_link_ang_vel_b_ta = TorchArray(self._root_link_ang_vel_b.data)
self._root_com_lin_vel_b_ta = TorchArray(self._root_com_lin_vel_b.data)
self._root_com_ang_vel_b_ta = TorchArray(self._root_com_ang_vel_b.data)

# Pin TorchArrays — Category C (pre-allocated constants)
self._default_root_pose_ta = TorchArray(self._default_root_pose)
self._default_root_vel_ta = TorchArray(self._default_root_vel)
self._default_joint_pos_ta = TorchArray(self._default_joint_pos)
self._default_joint_vel_ta = TorchArray(self._default_joint_vel)
self._joint_pos_target_ta = TorchArray(self._joint_pos_target)
self._joint_vel_target_ta = TorchArray(self._joint_vel_target)
self._joint_effort_target_ta = TorchArray(self._joint_effort_target)
self._computed_torque_ta = TorchArray(self._computed_torque)
self._applied_torque_ta = TorchArray(self._applied_torque)
self._joint_stiffness_ta = TorchArray(self._joint_stiffness)
self._joint_damping_ta = TorchArray(self._joint_damping)
self._joint_armature_ta = TorchArray(self._joint_armature)
self._joint_friction_coeff_ta = TorchArray(self._joint_friction_coeff)
self._joint_dynamic_friction_coeff_ta = TorchArray(self._joint_dynamic_friction_coeff)
self._joint_viscous_friction_coeff_ta = TorchArray(self._joint_viscous_friction_coeff)
self._joint_pos_limits_ta = TorchArray(self._joint_pos_limits)
self._joint_vel_limits_ta = TorchArray(self._joint_vel_limits)
self._joint_effort_limits_ta = TorchArray(self._joint_effort_limits)
self._soft_joint_pos_limits_ta = TorchArray(self._soft_joint_pos_limits)
self._soft_joint_vel_limits_ta = TorchArray(self._soft_joint_vel_limits)
self._gear_ratio_ta = TorchArray(self._gear_ratio)
self._body_mass_ta = TorchArray(self._body_mass)
self._body_inertia_ta = TorchArray(self._body_inertia)
# Tendons (conditional)
if self._num_fixed_tendons > 0:
    self._fixed_tendon_stiffness_ta = TorchArray(self._fixed_tendon_stiffness)
    self._fixed_tendon_damping_ta = TorchArray(self._fixed_tendon_damping)
    self._fixed_tendon_limit_stiffness_ta = TorchArray(self._fixed_tendon_limit_stiffness)
    self._fixed_tendon_rest_length_ta = TorchArray(self._fixed_tendon_rest_length)
    self._fixed_tendon_offset_ta = TorchArray(self._fixed_tendon_offset)
    self._fixed_tendon_pos_limits_ta = TorchArray(self._fixed_tendon_pos_limits)
if self._num_spatial_tendons > 0:
    self._spatial_tendon_stiffness_ta = TorchArray(self._spatial_tendon_stiffness)
    self._spatial_tendon_damping_ta = TorchArray(self._spatial_tendon_damping)
    self._spatial_tendon_limit_stiffness_ta = TorchArray(self._spatial_tendon_limit_stiffness)
    self._spatial_tendon_offset_ta = TorchArray(self._spatial_tendon_offset)

# Category D — lazy sliced (initialized to None)
self._root_link_pos_w_ta: TorchArray | None = None
self._root_link_quat_w_ta: TorchArray | None = None
# ... (all 20 sliced property pairs set to None, same as Newton pattern)
```

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
git -c commit.gpgsign=false commit -m "Add _create_simulation_bindings and refactor _create_buffers

Cache PhysX view getter results as direct simulation bindings.
Remove TimestampedBuffer for Category A properties. Pin TorchArrays
for all categories."
```

---

### Task 2: Simplify Category A properties to one-liner returns

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`

- [ ] **Step 1: Simplify all Category A properties**

For each Category A property, replace the timestamp-check-and-refresh pattern with a direct return of the pinned TorchArray:

```python
# root_link_pose_w — BEFORE:
@property
def root_link_pose_w(self) -> TorchArray:
    if self._root_link_pose_w.timestamp < self._sim_timestamp:
        self._root_link_pose_w.data = self._root_view.get_root_transforms().view(wp.transformf)
        self._root_link_pose_w.timestamp = self._sim_timestamp
    return TorchArray(self._root_link_pose_w.data)

# root_link_pose_w — AFTER:
@property
def root_link_pose_w(self) -> TorchArray:
    """Root link pose ..."""
    return self._root_link_pose_w_ta
```

Apply to ALL Category A properties:
- `root_link_pose_w` → `return self._root_link_pose_w_ta`
- `root_com_vel_w` → `return self._root_com_vel_w_ta`
- `body_link_pose_w` → `return self._body_link_pose_w_ta`
- `body_com_vel_w` → `return self._body_com_vel_w_ta`
- `body_com_acc_w` → `return self._body_com_acc_w_ta`
- `body_incoming_joint_wrench_b` → `return self._body_incoming_joint_wrench_b_ta`
- `joint_pos` → `return self._joint_pos_ta`
- `joint_vel` → `return self._joint_vel_ta`

- [ ] **Step 2: Update `body_com_pose_b` (Category A* — keeps timestamp for CPU→GPU copy)**

This property uses `get_coms()` which returns a CPU array needing `wp.copy()`. Keep the timestamp check but return a pinned TorchArray:

```python
@property
def body_com_pose_b(self) -> TorchArray:
    if self._body_com_pose_b.timestamp < self._sim_timestamp:
        self._body_com_pose_b.data.assign(self._root_view.get_coms().view(wp.transformf))
        self._body_com_pose_b.timestamp = self._sim_timestamp
    return self._body_com_pose_b_ta  # pinned, not TorchArray(self._body_com_pose_b.data)
```

- [ ] **Step 3: Update Category B properties to return pinned TorchArrays**

For each Category B property, change only the return statement:

```python
# BEFORE:
return TorchArray(self._root_link_vel_w.data)

# AFTER:
return self._root_link_vel_w_ta
```

Apply to: `root_link_vel_w`, `root_com_pose_w`, `body_link_vel_w`, `body_com_pose_w`, `projected_gravity_b`, `heading_w`, `root_link_lin_vel_b`, `root_link_ang_vel_b`, `root_com_lin_vel_b`, `root_com_ang_vel_b`, `joint_acc`, `root_state_w`, `root_link_state_w`, `root_com_state_w`, `body_state_w`, `body_link_state_w`, `body_com_state_w`.

- [ ] **Step 4: Update Category C properties to return pinned TorchArrays**

For each Category C property, change the return:

```python
# BEFORE:
return TorchArray(self._default_root_pose)

# AFTER:
return self._default_root_pose_ta
```

Apply to ALL Category C properties: `default_root_pose`, `default_root_vel`, `default_joint_pos`, `default_joint_vel`, `joint_pos_target`, `joint_vel_target`, `joint_effort_target`, `computed_torque`, `applied_torque`, `joint_stiffness`, `joint_damping`, `joint_armature`, `joint_friction_coeff`, `joint_dynamic_friction_coeff`, `joint_viscous_friction_coeff`, `joint_pos_limits`, `joint_vel_limits`, `joint_effort_limits`, `soft_joint_pos_limits`, `soft_joint_vel_limits`, `gear_ratio`, `body_mass`, `body_inertia`, all tendon properties.

- [ ] **Step 5: Update Category D sliced properties to use lazy pinning**

For each sliced property, change from dynamic TorchArray to lazy pin:

```python
# BEFORE:
@property
def root_link_pos_w(self) -> TorchArray:
    return TorchArray(self._get_pos_from_transform(self.root_link_pose_w.warp))

# AFTER:
@property
def root_link_pos_w(self) -> TorchArray:
    if self._root_link_pos_w_ta is None:
        self._root_link_pos_w_ta = TorchArray(
            self._get_pos_from_transform(self.root_link_pose_w.warp)
        )
    return self._root_link_pos_w_ta
```

Apply to ALL Category D properties (20 total — same list as in spec).

**IMPORTANT:** PhysX sliced helpers (`_get_pos_from_transform` etc.) always return contiguous strided views (pointer arithmetic). Unlike Newton, there's no non-contiguous kernel path. So the TorchArray wraps a stable view — pinning is safe.

- [ ] **Step 6: Update internal kernel inputs to use `_sim_bind_*` instead of property `.warp`**

Category B properties that call `wp.launch()` with Category A sibling properties as inputs should now use `self._sim_bind_*` directly instead of `self.property.warp`:

```python
# BEFORE (in root_link_vel_w):
inputs=[self.root_com_vel_w.warp, self.root_link_pose_w.warp, self.body_com_pose_b.warp]

# AFTER:
inputs=[self._sim_bind_root_com_vel_w, self._sim_bind_root_link_pose_w, self._body_com_pose_b.data]
```

This avoids going through the property accessor when we already have the raw array. Apply to all Category B kernels that reference Category A siblings.

- [ ] **Step 7: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
git -c commit.gpgsign=false commit -m "Simplify PhysX ArticulationData properties with pinned TorchArrays

Category A: one-liner returns (no timestamp check).
Category B: pinned TorchArray returns.
Category C: pinned TorchArray returns.
Category D: lazy pinned sliced properties."
```

---

### Task 3: Register PHYSICS_READY callback in Articulation

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`

- [ ] **Step 1: Add the PHYSICS_READY callback registration**

In `_initialize_impl()`, after `self._data = ArticulationData(...)` and before `self._create_buffers()`, add:

```python
# container for data access
self._data = ArticulationData(self.root_view, self.device)

# Register callback to rebind simulation data after a full reset.
from isaaclab.physics import PhysicsEvent
self._physics_ready_handle = SimulationManager.register_callback(
    lambda _: self._data._create_simulation_bindings(),
    PhysicsEvent.PHYSICS_READY,
    name=f"articulation_rebind_{self.cfg.prim_path}",
)

# create buffers
self._create_buffers()
```

Check if `PhysicsEvent` is already imported; if not, add the import at the top of the file.

- [ ] **Step 2: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py
git -c commit.gpgsign=false commit -m "Register PHYSICS_READY callback for PhysX ArticulationData rebind"
```

---

### Task 4: Run validation tests

**Files:** None modified — validation only.

- [ ] **Step 1: Run PhysX articulation asset tests (must be 210/210)**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/assets/test_articulation.py -v --tb=short 2>&1 | tail -5
```

Expected: `210 passed`

- [ ] **Step 2: Run mock view tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view_warp.py -v 2>&1 | tail -5
```

Expected: All pass (these test the view layer, not data properties).

- [ ] **Step 3: Run pre-commit**

```bash
./isaaclab.sh -f
```

- [ ] **Step 4: Fix any failures**

If tests fail:
- `TorchArray` attribute errors → missing pin or rebind
- `wp.array` type errors → forgot `.warp` on a kernel input
- Numerical mismatches → stale data, check if a property needs the timestamp pattern kept

Fix, re-run, iterate until green.

- [ ] **Step 5: Commit fixes if any**

```bash
git add -u
git -c commit.gpgsign=false commit -m "Fix test failures from PhysX simulation bindings refactor"
```
