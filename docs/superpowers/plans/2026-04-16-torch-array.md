# TorchArray Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TorchArray` wrapper so articulation data properties provide explicit `.torch` and `.warp` accessors instead of raw `wp.array`, eliminating the need for `wp.to_torch()` everywhere.

**Architecture:** A new `TorchArray` class in `isaaclab.utils.warp` wraps a `wp.array` and provides cached zero-copy `.torch` / `.warp` accessors. All `@property` methods in `BaseArticulationData`, the PhysX `ArticulationData`, and the Newton `ArticulationData` are updated to return `TorchArray`. A deprecation bridge (`__torch_function__` + arithmetic operators) provides backwards compatibility during migration.

**Tech Stack:** Python, warp (`wp`), PyTorch (`torch`), pytest

**Spec:** `docs/superpowers/specs/2026-04-16-torch-array-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `source/isaaclab/isaaclab/utils/warp/torch_array.py` | `TorchArray` class |
| Modify | `source/isaaclab/isaaclab/utils/warp/__init__.pyi` | Export `TorchArray` |
| Create | `source/isaaclab/test/utils/warp/test_torch_array.py` | Unit tests for `TorchArray` |
| Modify | `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py` | Return type hints `wp.array` → `TorchArray` |
| Modify | `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py` | Wrap returns in `TorchArray()` |
| Modify | `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py` | Wrap returns in `TorchArray()` |
| Modify | `source/isaaclab/config/extension.toml` | Bump version `4.6.1` → `4.6.2` |
| Modify | `source/isaaclab/docs/CHANGELOG.rst` | Add changelog entry |
| Modify | `source/isaaclab_physx/config/extension.toml` | Bump version `0.5.16` → `0.5.17` |
| Modify | `source/isaaclab_physx/docs/CHANGELOG.rst` | Add changelog entry |
| Modify | `source/isaaclab_newton/config/extension.toml` | Bump version `0.5.13` → `0.5.14` |
| Modify | `source/isaaclab_newton/docs/CHANGELOG.rst` | Add changelog entry |

---

### Task 1: Create the `TorchArray` class with tests (TDD)

**Files:**
- Create: `source/isaaclab/isaaclab/utils/warp/torch_array.py`
- Create: `source/isaaclab/test/utils/warp/test_torch_array.py`
- Modify: `source/isaaclab/isaaclab/utils/warp/__init__.pyi`

- [ ] **Step 1: Write the test file**

Create `source/isaaclab/test/utils/warp/test_torch_array.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TorchArray."""

import warnings

import pytest
import torch
import warp as wp


class TestTorchArrayBasic:
    """Tests for TorchArray core accessors."""

    def test_warp_returns_original_array(self):
        """Test that .warp returns the original wp.array."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(10, dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.warp is wp_arr

    def test_torch_returns_tensor(self):
        """Test that .torch returns a torch.Tensor."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(10, dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        result = ta.torch
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)

    def test_torch_is_cached(self):
        """Test that .torch returns the same object on repeated access."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(10, dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        first = ta.torch
        second = ta.torch
        assert first is second

    def test_torch_shares_memory(self):
        """Test that .torch is a zero-copy view (shared memory)."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        ta.torch[0] = 99.0
        assert wp_arr.numpy()[0] == 99.0


class TestTorchArrayStructuredTypes:
    """Tests for TorchArray with structured warp types."""

    def test_vec3f(self):
        """Test .torch expands vec3f to (N, 3) float32."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.warp.shape == (5,)
        assert ta.warp.dtype == wp.vec3f
        assert ta.torch.shape == (5, 3)
        assert ta.torch.dtype == torch.float32

    def test_quatf(self):
        """Test .torch expands quatf to (N, 4) float32."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.quatf, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.torch.shape == (5, 4)

    def test_transformf(self):
        """Test .torch expands transformf to (N, 7) float32."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.transformf, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.torch.shape == (5, 7)

    def test_spatial_vectorf(self):
        """Test .torch expands spatial_vectorf to (N, 6) float32."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.spatial_vectorf, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.torch.shape == (5, 6)

    def test_2d_vec3f(self):
        """Test .torch expands 2D vec3f to (N, M, 3) float32."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros((4, 13), dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.torch.shape == (4, 13, 3)


class TestTorchArrayConvenienceProperties:
    """Tests for shape, dtype, device, len, repr."""

    def test_shape(self):
        """Test shape returns warp shape (not torch shape)."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.shape == (5,)

    def test_dtype(self):
        """Test dtype returns warp dtype."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        assert ta.dtype == wp.vec3f

    def test_device(self):
        """Test device returns warp device string."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        assert "cpu" in str(ta.device)

    def test_len(self):
        """Test len returns first dimension."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros((4, 12), dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)
        assert len(ta) == 4

    def test_repr(self):
        """Test repr is informative."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.zeros(5, dtype=wp.vec3f, device="cpu")
        ta = TorchArray(wp_arr)
        r = repr(ta)
        assert "TorchArray" in r
        assert "vec3f" in r


class TestTorchArrayDeprecationBridge:
    """Tests for __torch_function__ and arithmetic operators with deprecation warnings."""

    def setup_method(self):
        """Reset deprecation warning state before each test."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = False

    def test_torch_function_works_and_warns(self):
        """Test torch.* functions work on TorchArray but emit deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        wp_arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        ta = TorchArray(wp_arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.sum(ta)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert ".torch" in str(w[0].message)

        assert isinstance(result, torch.Tensor)
        assert result.item() == 6.0

    def test_torch_cat_works_and_warns(self):
        """Test torch.cat works with TorchArray."""
        from isaaclab.utils.warp.torch_array import TorchArray

        a = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        b = TorchArray(wp.array([3.0, 4.0], dtype=wp.float32, device="cpu"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cat([a, b])
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_add_scalar_works_and_warns(self):
        """Test TorchArray + scalar works with deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ta + 1.0
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([2.0, 3.0]))

    def test_radd_works(self):
        """Test scalar + TorchArray works."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True  # Suppress warning for simplicity.
        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        result = 1.0 + ta
        assert torch.allclose(result, torch.tensor([2.0, 3.0]))

    def test_sub_works(self):
        """Test TorchArray - value works."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([3.0, 4.0], dtype=wp.float32, device="cpu"))
        result = ta - 1.0
        assert torch.allclose(result, torch.tensor([2.0, 3.0]))

    def test_mul_works(self):
        """Test TorchArray * value works."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([2.0, 3.0], dtype=wp.float32, device="cpu"))
        result = ta * 2.0
        assert torch.allclose(result, torch.tensor([4.0, 6.0]))

    def test_neg_works(self):
        """Test -TorchArray works."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, -2.0], dtype=wp.float32, device="cpu"))
        result = -ta
        assert torch.allclose(result, torch.tensor([-1.0, 2.0]))

    def test_comparison_works(self):
        """Test comparison operators."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"))
        result = ta > 1.5
        assert result.tolist() == [False, True, True]

    def test_deprecation_warns_only_once(self):
        """Test that the deprecation warning is emitted only once."""
        from isaaclab.utils.warp.torch_array import TorchArray

        ta = TorchArray(wp.array([1.0], dtype=wp.float32, device="cpu"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta + 1.0  # First: warns.
            _ = ta * 2.0  # Second: no warning.
            _ = torch.sum(ta)  # Third: no warning.
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1

    def test_torch_tensor_plus_torch_array(self):
        """Test torch.Tensor + TorchArray works via __torch_function__."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = True
        ta = TorchArray(wp.array([1.0, 2.0], dtype=wp.float32, device="cpu"))
        t = torch.tensor([10.0, 20.0])
        result = t + ta
        assert torch.allclose(result, torch.tensor([11.0, 22.0]))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/utils/warp/test_torch_array.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError` because `torch_array.py` doesn't exist yet.

- [ ] **Step 3: Implement `TorchArray`**

Create `source/isaaclab/isaaclab/utils/warp/torch_array.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first dual-access array with explicit torch interoperability.

Attribution: Inspired by TorchArray from mujocolab/mjlab
(https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sim/sim_data.py, BSD-3-Clause).
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import torch
import warp as wp


class TorchArray:
    """Warp array with explicit torch interoperability.

    Provides two explicit accessors:

    - ``.torch`` — cached zero-copy :class:`torch.Tensor` view.
    - ``.warp`` — the underlying :class:`wp.array`.

    During the deprecation period, implicit torch operations (arithmetic,
    :meth:`__torch_function__`) work but emit a one-time warning directing
    users to ``.torch``.

    Example:

    .. code-block:: python

        # Explicit torch access (preferred):
        pos = robot.data.root_link_pos_w.torch

        # Explicit warp access (for kernels):
        wp.launch(kernel, inputs=[robot.data.root_link_pos_w.warp], ...)

        # Implicit torch (deprecated — emits warning):
        scaled = robot.data.joint_pos * 2.0
    """

    _deprecation_warned: ClassVar[bool] = False

    def __init__(self, wp_array: wp.array) -> None:
        self._warp = wp_array
        self._torch_cache: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Explicit accessors
    # ------------------------------------------------------------------

    @property
    def torch(self) -> torch.Tensor:
        """Cached zero-copy :class:`torch.Tensor` view of the underlying warp array.

        For structured warp types the trailing component dimension is expanded
        (e.g. ``(N,) vec3f`` becomes ``(N, 3) float32``).
        """
        if self._torch_cache is None:
            self._torch_cache = wp.to_torch(self._warp)
        return self._torch_cache

    @property
    def warp(self) -> wp.array:
        """The underlying :class:`wp.array`."""
        return self._warp

    # ------------------------------------------------------------------
    # Convenience (no deprecation — informational)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """Shape of the warp array (structured-type dimensions not expanded)."""
        return self._warp.shape

    @property
    def dtype(self):
        """Warp dtype of the underlying array."""
        return self._warp.dtype

    @property
    def device(self):
        """Device of the underlying warp array."""
        return self._warp.device

    def __len__(self) -> int:
        return self._warp.shape[0]

    def __repr__(self) -> str:
        return f"TorchArray(shape={self._warp.shape}, dtype={self._warp.dtype}, device={self._warp.device})"

    # ------------------------------------------------------------------
    # Deprecation bridge (to be removed in a future release)
    # ------------------------------------------------------------------

    @classmethod
    def _warn_implicit(cls) -> None:
        """Emit a one-time deprecation warning for implicit torch conversion."""
        if not cls._deprecation_warned:
            warnings.warn(
                "Implicit torch conversion of TorchArray is deprecated. "
                "Use .torch for torch.Tensor or .warp for wp.array explicitly.",
                DeprecationWarning,
                stacklevel=3,
            )
            cls._deprecation_warned = True

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Intercept ``torch.*`` function calls, unwrap to ``.torch``, and warn."""
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented
        cls._warn_implicit()

        def _unwrap(x: Any) -> Any:
            return x.torch if isinstance(x, cls) else x

        unwrapped_args = tuple(_unwrap(a) for a in args)
        unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return func(*unwrapped_args, **unwrapped_kwargs)

    # Arithmetic operators

    def __add__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch + (other.torch if isinstance(other, TorchArray) else other)

    def __radd__(self, other: Any) -> Any:
        self._warn_implicit()
        return other + self.torch

    def __sub__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch - (other.torch if isinstance(other, TorchArray) else other)

    def __rsub__(self, other: Any) -> Any:
        self._warn_implicit()
        return other - self.torch

    def __mul__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch * (other.torch if isinstance(other, TorchArray) else other)

    def __rmul__(self, other: Any) -> Any:
        self._warn_implicit()
        return other * self.torch

    def __truediv__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch / (other.torch if isinstance(other, TorchArray) else other)

    def __rtruediv__(self, other: Any) -> Any:
        self._warn_implicit()
        return other / self.torch

    def __pow__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch ** (other.torch if isinstance(other, TorchArray) else other)

    def __rpow__(self, other: Any) -> Any:
        self._warn_implicit()
        return other**self.torch

    def __neg__(self) -> Any:
        self._warn_implicit()
        return -self.torch

    def __pos__(self) -> Any:
        self._warn_implicit()
        return +self.torch

    def __abs__(self) -> Any:
        self._warn_implicit()
        return abs(self.torch)

    # Comparison operators

    def __eq__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch == (other.torch if isinstance(other, TorchArray) else other)

    def __ne__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch != (other.torch if isinstance(other, TorchArray) else other)

    def __lt__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch < (other.torch if isinstance(other, TorchArray) else other)

    def __le__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch <= (other.torch if isinstance(other, TorchArray) else other)

    def __gt__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch > (other.torch if isinstance(other, TorchArray) else other)

    def __ge__(self, other: Any) -> Any:
        self._warn_implicit()
        return self.torch >= (other.torch if isinstance(other, TorchArray) else other)
```

- [ ] **Step 4: Update the `__init__.pyi` stub**

In `source/isaaclab/isaaclab/utils/warp/__init__.pyi`, add the `TorchArray` export:

```python
__all__ = [
    "TorchArray",
    "convert_to_warp_mesh",
    "raycast_dynamic_meshes",
    "raycast_mesh",
    "raycast_single_mesh",
]

from .ops import convert_to_warp_mesh, raycast_dynamic_meshes, raycast_mesh, raycast_single_mesh
from .torch_array import TorchArray
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/utils/warp/test_torch_array.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add source/isaaclab/isaaclab/utils/warp/torch_array.py \
       source/isaaclab/isaaclab/utils/warp/__init__.pyi \
       source/isaaclab/test/utils/warp/test_torch_array.py
git commit -m "Add TorchArray class with explicit .torch/.warp accessors

Warp-first dual-access array wrapper. Provides cached zero-copy
.torch view and .warp accessor for kernel interop. Includes a
deprecation bridge (__torch_function__ + operators) for migration."
```

---

### Task 2: Update `BaseArticulationData` type hints

**Files:**
- Modify: `source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py`

This file has ~119 properties returning `-> wp.array`. All return type hints must change to `-> TorchArray`. The implementations in the base class are either `@abstractmethod` (no body change needed), shorthands (delegate to another property — no body change needed), or deprecated wrappers (return a `wp.array` via `wp.clone` — these need wrapping too).

- [ ] **Step 1: Add import and change all abstract property return types**

At the top of the file, add the import:

```python
import warnings
from abc import ABC, abstractmethod

import warp as wp

from isaaclab.utils.warp import TorchArray
```

Then change **every** property signature from `-> wp.array:` to `-> TorchArray:`. This is a bulk find-and-replace across the file. There are 119 occurrences of `-> wp.array:` to change.

**Important:** Do NOT change method parameters that accept `wp.array` (e.g. setter signatures like `def default_root_pose(self, value: wp.array)`). Only change return types.

- [ ] **Step 2: Wrap the shorthand property returns**

The shorthand properties (lines ~991-1109) delegate to other properties. Since those other properties will now return `TorchArray`, the shorthands automatically return `TorchArray` — **no body changes needed**. Only the type hint changes from step 1.

- [ ] **Step 3: Wrap the deprecated property returns**

The deprecated `default_*` properties (lines ~1115-1377) return raw `wp.array` via `wp.clone()`. These need wrapping. Example pattern:

```python
@property
def default_mass(self) -> TorchArray:
    """Deprecated property. Please use :attr:`body_mass` instead..."""
    warnings.warn(...)
    if self._default_mass is None:
        self._default_mass = wp.clone(self.body_mass.warp, self.device)
    return TorchArray(self._default_mass)
```

Note: `self.body_mass` now returns `TorchArray`, so `wp.clone` needs `.warp`. Apply this pattern to all deprecated properties that call `.warp` on a sibling property:

- `default_mass` (uses `self.body_mass`)
- `default_inertia` (uses `self.body_inertia`)
- `default_joint_stiffness` (uses `self.joint_stiffness`)
- `default_joint_damping` (uses `self.joint_damping`)
- `default_joint_armature` (uses `self.joint_armature`)
- `default_joint_friction_coeff` (uses `self.joint_friction_coeff`)
- `default_joint_viscous_friction_coeff` — check if it references another property
- `default_joint_pos_limits` (uses `self.joint_pos_limits`)
- `default_fixed_tendon_stiffness` through `default_fixed_tendon_pos_limits` (uses corresponding tendon properties)
- `default_spatial_tendon_stiffness` through `default_spatial_tendon_offset` (uses corresponding spatial tendon properties)
- `default_fixed_tendon_limit` (shorthand for `default_fixed_tendon_pos_limits`)
- `default_joint_friction` (shorthand for `default_joint_friction_coeff`)

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py
git commit -m "Update BaseArticulationData return types to TorchArray

Change all property return type hints from wp.array to TorchArray.
Wrap deprecated property returns in TorchArray()."
```

---

### Task 3: Update PhysX `ArticulationData` to return `TorchArray`

**Files:**
- Modify: `source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py`

This file has ~81 properties. Each one that returns a `wp.array` needs to wrap the return value in `TorchArray()`. Internal warp kernel code (`wp.launch()`) is unchanged — it uses raw `self._buffer.data`.

- [ ] **Step 1: Add import**

At the top of the file, add:

```python
from isaaclab.utils.warp import TorchArray
```

- [ ] **Step 2: Update return types and wrap buffered properties**

Buffered properties (those using `TimestampedBufferWarp`) follow this pattern. Change each one:

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

Apply to all buffered properties. The complete list (from the PhysX file):

**Defaults:** `default_root_pose`, `default_root_vel`, `default_joint_pos`, `default_joint_vel`

**Joint commands:** `joint_pos_target`, `joint_vel_target`, `joint_effort_target`

**Actuator outputs:** `computed_torque`, `applied_torque`

**Joint properties:** `joint_stiffness`, `joint_damping`, `joint_armature`, `joint_friction_coeff`, `joint_pos_limits`, `joint_vel_limits`, `joint_effort_limits`, `soft_joint_pos_limits`, `soft_joint_vel_limits`, `gear_ratio`

**Tendon properties:** `fixed_tendon_stiffness`, `fixed_tendon_damping`, `fixed_tendon_limit_stiffness`, `fixed_tendon_rest_length`, `fixed_tendon_offset`, `fixed_tendon_pos_limits`, `spatial_tendon_stiffness`, `spatial_tendon_damping`, `spatial_tendon_limit_stiffness`, `spatial_tendon_offset`

**Root state:** `root_link_pose_w`, `root_link_vel_w`, `root_com_pose_w`, `root_com_vel_w`

**Body state:** `body_mass`, `body_inertia`, `body_link_pose_w`, `body_link_vel_w`, `body_com_pose_w`, `body_com_vel_w`, `body_com_acc_w`, `body_com_pose_b`, `body_incoming_joint_wrench_b`

**Joint state:** `joint_pos`, `joint_vel`, `joint_acc`

**Derived:** `projected_gravity_b`, `heading_w`, `root_link_lin_vel_b`, `root_link_ang_vel_b`, `root_com_lin_vel_b`, `root_com_ang_vel_b`

- [ ] **Step 3: Update sliced properties**

Sliced properties call internal helpers that expect `wp.array`. They need `.warp` on the parent TorchArray:

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

Apply to all sliced properties:

- `root_link_pos_w`, `root_link_quat_w` (from `root_link_pose_w`)
- `root_link_lin_vel_w`, `root_link_ang_vel_w` (from `root_link_vel_w`)
- `root_com_pos_w`, `root_com_quat_w` (from `root_com_pose_w`)
- `root_com_lin_vel_w`, `root_com_ang_vel_w` (from `root_com_vel_w`)
- `body_link_pos_w`, `body_link_quat_w` (from `body_link_pose_w`)
- `body_link_lin_vel_w`, `body_link_ang_vel_w` (from `body_link_vel_w`)
- `body_com_pos_w`, `body_com_quat_w` (from `body_com_pose_w`)
- `body_com_lin_vel_w`, `body_com_ang_vel_w` (from `body_com_vel_w`)
- `body_com_lin_acc_w`, `body_com_ang_acc_w` (from `body_com_acc_w`)
- `body_com_pos_b`, `body_com_quat_b` (from `body_com_pose_b`)

- [ ] **Step 4: Update derived properties that reference sibling properties in `wp.launch()`**

Derived properties like `projected_gravity_b` pass sibling property results to `wp.launch()`. Since sibling properties now return `TorchArray`, the inputs to `wp.launch()` need `.warp`:

```python
# Before:
@property
def projected_gravity_b(self):
    if self._projected_gravity_b.timestamp < self._sim_timestamp:
        wp.launch(
            shared_kernels.quat_apply_inverse_1D_kernel,
            dim=self._num_instances,
            inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w],
            outputs=[self._projected_gravity_b.data],
            device=self.device,
        )
        self._projected_gravity_b.timestamp = self._sim_timestamp
    return self._projected_gravity_b.data

# After:
@property
def projected_gravity_b(self) -> TorchArray:
    if self._projected_gravity_b.timestamp < self._sim_timestamp:
        wp.launch(
            shared_kernels.quat_apply_inverse_1D_kernel,
            dim=self._num_instances,
            inputs=[self.GRAVITY_VEC_W, self.root_link_quat_w.warp],
            outputs=[self._projected_gravity_b.data],
            device=self.device,
        )
        self._projected_gravity_b.timestamp = self._sim_timestamp
    return TorchArray(self._projected_gravity_b.data)
```

Apply this pattern to all derived properties that reference sibling properties in `wp.launch()` inputs:

- `projected_gravity_b` — uses `self.root_link_quat_w`
- `heading_w` — uses `self.root_link_quat_w`
- `root_link_lin_vel_b` — uses `self.root_link_lin_vel_w`, `self.root_link_quat_w`
- `root_link_ang_vel_b` — uses `self.root_link_ang_vel_w`, `self.root_link_quat_w`
- `root_com_lin_vel_b` — uses `self.root_com_lin_vel_w`, `self.root_link_quat_w`
- `root_com_ang_vel_b` — uses `self.root_com_ang_vel_w`, `self.root_link_quat_w`
- `joint_acc` — uses `self.joint_vel` (which is now TorchArray, passed to wp.launch)

For `joint_acc`, the `self.joint_vel` reference in the `wp.launch()` inputs list needs `.warp`:
```python
inputs=[
    self.joint_vel.warp,
    self._previous_joint_vel,
    time_elapsed,
],
```

- [ ] **Step 5: Update the `update()` method**

The `update()` method calls `self.joint_acc` to trigger lazy refresh. This still works since accessing the property triggers the side-effect. No change needed.

- [ ] **Step 6: Update deprecated state properties**

The deprecated `root_state_w`, `root_link_state_w`, `root_com_state_w`, `body_state_w`, `body_link_state_w`, `body_com_state_w` likely concatenate other properties. Check if they call `wp.to_torch()` or warp operations on sibling properties, and update accordingly. The pattern is the same: sibling properties accessed in warp ops need `.warp`, and the final return wraps in `TorchArray()`.

- [ ] **Step 7: Commit**

```bash
git add source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py
git commit -m "Update PhysX ArticulationData to return TorchArray

Wrap all property returns in TorchArray(). Internal warp kernels
access sibling properties via .warp for wp.launch() compatibility."
```

---

### Task 4: Update Newton `ArticulationData` to return `TorchArray`

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py`

Same pattern as PhysX but with Newton-specific differences:
- Newton sliced helpers take a `source` cache buffer as first arg
- Newton checks `transform.is_contiguous` for strided vs kernel path
- Newton uses `_sim_bind_*` for direct simulation bindings

- [ ] **Step 1: Add import**

```python
from isaaclab.utils.warp import TorchArray
```

- [ ] **Step 2: Update return types and wrap buffered properties**

Same pattern as PhysX Task 3, Step 2. Change return type to `TorchArray`, wrap `return` in `TorchArray()`.

- [ ] **Step 3: Update sliced properties**

Newton sliced properties pass a cache buffer and the parent property to the helper:

```python
# Before:
@property
def root_link_pos_w(self) -> wp.array:
    return self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w)

# After:
@property
def root_link_pos_w(self) -> TorchArray:
    return TorchArray(self._get_pos_from_transform(self._root_link_pos_w, self.root_link_pose_w.warp))
```

The second argument (`self.root_link_pose_w`) now returns `TorchArray`, so it needs `.warp` since the helper expects `wp.array`. Apply to all Newton sliced properties (same list as PhysX Task 3, Step 3).

- [ ] **Step 4: Update derived properties**

Same pattern as PhysX Task 3, Step 4. Add `.warp` to sibling property references in `wp.launch()` inputs.

- [ ] **Step 5: Update deprecated state properties**

Same as PhysX Task 3, Step 6.

- [ ] **Step 6: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py
git commit -m "Update Newton ArticulationData to return TorchArray

Same pattern as PhysX: wrap returns in TorchArray(), access sibling
properties via .warp for wp.launch() compatibility."
```

---

### Task 5: Run validation tests

**Files:** None modified — validation only.

- [ ] **Step 1: Run TorchArray unit tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/utils/warp/test_torch_array.py -v
```

Expected: All PASS.

- [ ] **Step 2: Run PhysX mock articulation view tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view.py -v
```

Expected: All PASS. These tests use `wp.to_torch()` on view getters (not data properties), so they should be unaffected.

- [ ] **Step 3: Run PhysX mock articulation view warp tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/test_mock_interfaces/test_mock_articulation_view_warp.py -v
```

Expected: All PASS.

- [ ] **Step 4: Run Newton mock articulation view tests**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/test_mock_interfaces/test_mock_articulation_view.py -v
```

Expected: All PASS.

- [ ] **Step 5: Run pre-commit checks**

```bash
./isaaclab.sh -f
```

Expected: All checks pass. If formatting changes are needed, stage the modified files and re-run.

- [ ] **Step 6: Commit any pre-commit fixes**

If pre-commit made formatting changes:

```bash
git add -u
git commit -m "Apply pre-commit formatting fixes"
```

---

### Task 6: Update changelogs and version bumps

**Files:**
- Modify: `source/isaaclab/config/extension.toml` (version `4.6.1` → `4.6.2`)
- Modify: `source/isaaclab/docs/CHANGELOG.rst`
- Modify: `source/isaaclab_physx/config/extension.toml` (version `0.5.16` → `0.5.17`)
- Modify: `source/isaaclab_physx/docs/CHANGELOG.rst`
- Modify: `source/isaaclab_newton/config/extension.toml` (version `0.5.13` → `0.5.14`)
- Modify: `source/isaaclab_newton/docs/CHANGELOG.rst`

- [ ] **Step 1: Bump isaaclab version and add changelog**

In `source/isaaclab/config/extension.toml`, change `version = "4.6.1"` to `version = "4.6.2"`.

Prepend to `source/isaaclab/docs/CHANGELOG.rst` (after the header):

```rst
4.6.2 (2026-04-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.utils.warp.TorchArray`, a warp-first dual-access array
  that provides explicit ``.torch`` and ``.warp`` accessors for seamless
  interoperability between warp and PyTorch workflows.

Changed
^^^^^^^

* All :class:`~isaaclab.assets.articulation.BaseArticulationData` properties now
  return :class:`~isaaclab.utils.warp.TorchArray` instead of raw ``wp.array``.
  Use ``.torch`` for a cached zero-copy ``torch.Tensor`` view, or ``.warp`` for
  the underlying ``wp.array``. Implicit torch operations (arithmetic,
  ``torch.*`` functions) work during the deprecation period but emit a warning.


```

- [ ] **Step 2: Bump isaaclab_physx version and add changelog**

In `source/isaaclab_physx/config/extension.toml`, change `version = "0.5.16"` to `version = "0.5.17"`.

Prepend to `source/isaaclab_physx/docs/CHANGELOG.rst` (after the header):

```rst
0.5.17 (2026-04-16)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :class:`~isaaclab_physx.assets.articulation.ArticulationData` properties now
  return :class:`~isaaclab.utils.warp.TorchArray` instead of raw ``wp.array``.


```

- [ ] **Step 3: Bump isaaclab_newton version and add changelog**

In `source/isaaclab_newton/config/extension.toml`, change `version = "0.5.13"` to `version = "0.5.14"`.

Prepend to `source/isaaclab_newton/docs/CHANGELOG.rst` (after the header):

```rst
0.5.14 (2026-04-16)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :class:`~isaaclab_newton.assets.articulation.ArticulationData` properties now
  return :class:`~isaaclab.utils.warp.TorchArray` instead of raw ``wp.array``.


```

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab/config/extension.toml \
       source/isaaclab/docs/CHANGELOG.rst \
       source/isaaclab_physx/config/extension.toml \
       source/isaaclab_physx/docs/CHANGELOG.rst \
       source/isaaclab_newton/config/extension.toml \
       source/isaaclab_newton/docs/CHANGELOG.rst
git commit -m "Add changelog entries and bump versions for TorchArray"
```
