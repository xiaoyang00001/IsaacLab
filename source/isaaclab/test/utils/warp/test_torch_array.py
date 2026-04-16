# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TorchArray class."""

import warnings

import pytest
import torch
import warp as wp

wp.config.quiet = True
wp.init()


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Parametrize tests across CPU and CUDA devices."""
    return request.param


class TestTorchArrayBasic:
    """Tests for basic TorchArray functionality."""

    def test_warp_returns_original(self, device):
        """Test that .warp returns the original warp array."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.warp is arr

    def test_torch_returns_tensor(self, device):
        """Test that .torch returns a torch.Tensor."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert isinstance(ta.torch, torch.Tensor)

    def test_torch_is_cached(self, device):
        """Test that .torch returns the same tensor object on repeated access."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        t1 = ta.torch
        t2 = ta.torch
        assert t1 is t2

    def test_torch_shares_memory(self, device):
        """Test that .torch provides a zero-copy view (shares memory with warp)."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        t = ta.torch
        # Modify the torch tensor
        t[0] = 42.0
        # The change should be visible in the warp array
        arr_np = arr.numpy()
        assert arr_np[0] == 42.0


class TestTorchArrayStructuredTypes:
    """Tests for TorchArray with structured warp types (vec3f, quatf, etc)."""

    def test_vec3f_shape(self, device):
        """Test that vec3f arrays produce (N, 3) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.vec3f, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 3)

    def test_quatf_shape(self, device):
        """Test that quatf arrays produce (N, 4) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.quatf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 4)

    def test_transformf_shape(self, device):
        """Test that transformf arrays produce (N, 7) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.transformf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 7)

    def test_spatial_vectorf_shape(self, device):
        """Test that spatial_vectorf arrays produce (N, 6) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(8, dtype=wp.spatial_vectorf, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (8, 6)

    def test_2d_vec3f_shape(self, device):
        """Test that 2D vec3f arrays produce (N, M, 3) torch tensors."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((4, 5), dtype=wp.vec3f, device=device)
        ta = TorchArray(arr)
        assert ta.torch.shape == (4, 5, 3)


class TestTorchArrayConvenienceProperties:
    """Tests for convenience properties: shape, dtype, device, len, repr."""

    def test_shape(self, device):
        """Test that .shape returns the warp array shape."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((3, 4), dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.shape == (3, 4)

    def test_dtype(self, device):
        """Test that .dtype returns the warp dtype."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.dtype == wp.float32

    def test_device(self, device):
        """Test that .device returns the warp device string."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(10, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert ta.device == arr.device

    def test_len(self, device):
        """Test that len() returns the first dimension size."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros((7, 3), dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        assert len(ta) == 7

    def test_repr(self, device):
        """Test that repr() contains TorchArray and key info."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.zeros(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        r = repr(ta)
        assert "TorchArray" in r
        assert "float32" in r


class TestTorchArrayDeprecationBridge:
    """Tests for the deprecation bridge: __torch_function__, operators."""

    def setup_method(self):
        """Reset the deprecation warning flag before each test."""
        from isaaclab.utils.warp.torch_array import TorchArray

        TorchArray._deprecation_warned = False

    def test_torch_function_works_and_warns(self, device):
        """Test that __torch_function__ enables torch ops and emits a deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.sum(ta)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(result, torch.Tensor)
            assert result.item() == pytest.approx(5.0)

    def test_torch_cat_works_and_warns(self, device):
        """Test that torch.cat works with TorchArray and emits a deprecation warning."""
        from isaaclab.utils.warp.torch_array import TorchArray

        a1 = wp.ones(3, dtype=wp.float32, device=device)
        a2 = wp.ones(4, dtype=wp.float32, device=device)
        ta1, ta2 = TorchArray(a1), TorchArray(a2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cat([ta1, ta2])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result.shape == (7,)

    def test_arithmetic_operators_work_and_warn(self, device):
        """Test that arithmetic operators work and emit deprecation warnings."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ta + 1.0
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert isinstance(result, torch.Tensor)
            expected = torch.full((5,), 2.0, device=device)
            torch.testing.assert_close(result, expected)

    def test_warns_only_once(self, device):
        """Test that the deprecation warning is emitted only once per session."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = ta + 1.0
            _ = ta * 2.0
            _ = ta - 0.5
            # Only one warning despite three operations
            assert len(w) == 1

    def test_tensor_plus_torch_array(self, device):
        """Test that torch.Tensor + TorchArray works via __torch_function__."""
        from isaaclab.utils.warp.torch_array import TorchArray

        arr = wp.ones(5, dtype=wp.float32, device=device)
        ta = TorchArray(arr)
        t = torch.ones(5, device=device) * 2.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = t + ta
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            expected = torch.full((5,), 3.0, device=device)
            torch.testing.assert_close(result, expected)
