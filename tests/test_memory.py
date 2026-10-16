# Owner(s): ["module: PrivateUse1"]

import gc
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo

import torch_mcpu  # noqa: F401


class TestDeviceAllocator(TestCase):
    """Test cases for OpenRegDeviceAllocator functionality."""

    def setUp(self):
        """Reset memory state before each test."""
        # Force garbage collection to ensure clean state
        gc.collect()
        # Note: We can't directly reset allocator stats without C++ API,
        # but we can ensure tensors are properly released

    def test_basic_allocation(self):
        """Test basic memory allocation with various sizes."""
        # Small allocation
        x = torch.empty(100, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.numel(), 100)
        # Large allocation
        z = torch.empty(10000, device="mcpu")
        self.assertEqual(z.device.type, "mcpu")
        self.assertEqual(z.numel(), 10000)
        # Multi-dimensional allocation
        w = torch.empty(10, 20, 30, device="mcpu")
        self.assertEqual(w.device.type, "mcpu")
        self.assertEqual(w.shape, torch.Size([10, 20, 30]))

    def test_memory_lifecycle(self):
        """Test complete memory allocation and deallocation lifecycle."""
        # Allocate tensor
        x = torch.empty(1000, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")

        # Explicitly delete tensor
        del x
        gc.collect()

        # Allocate again to ensure memory was freed
        y = torch.empty(1000, device="mcpu")
        self.assertEqual(y.device.type, "mcpu")
        del y
        gc.collect()

    def test_tensor_copy_operations(self):
        """Test memory operations during tensor copies."""
        # CPU to OpenReg
        cpu_tensor = torch.randn(100)
        mcpu_tensor = cpu_tensor.to("mcpu")
        self.assertEqual(mcpu_tensor.device.type, "mcpu")
        self.assertEqual(cpu_tensor.shape, mcpu_tensor.shape)

        # OpenReg to CPU
        back_to_cpu = mcpu_tensor.to("cpu")
        self.assertEqual(back_to_cpu.device.type, "cpu")
        self.assertTrue(torch.allclose(cpu_tensor, back_to_cpu))

        # OpenReg to OpenReg (clone)
        cloned = mcpu_tensor.clone()
        self.assertEqual(cloned.device.type, "mcpu")
        self.assertTrue(torch.allclose(mcpu_tensor.cpu(), cloned.cpu()))

    def test_inplace_operations(self):
        """Test memory stability during inplace operations."""
        x = torch.ones(100, device="mcpu")
        original_data_ptr = x.data_ptr()

        # Inplace addition
        x.add_(1)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 2))

        # Inplace multiplication
        x.mul_(2)
        self.assertEqual(x.data_ptr(), original_data_ptr)
        self.assertTrue(torch.all(x == 4))

    def test_view_operations(self):
        """Test that views share memory correctly."""
        x = torch.randn(100, device="mcpu")
        original_data_ptr = x.data_ptr()

        # Reshape view
        y = x.view(10, 10)
        self.assertEqual(y.data_ptr(), original_data_ptr)
        self.assertEqual(y.shape, torch.Size([10, 10]))

        # Slice view
        z = x[10:20]
        # Slices may have different data_ptr but should share storage
        self.assertEqual(z.numel(), 10)

    def test_different_dtypes(self):
        """Test allocation with different data types."""
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.empty(100, dtype=dtype, device="mcpu")
            self.assertEqual(x.device.type, "mcpu")
            self.assertEqual(x.dtype, dtype)
            self.assertEqual(x.numel(), 100)

    def test_tensor_resize(self):
        """Test tensor resizing operations."""
        x = torch.empty(100, device="mcpu")
        _ = x.data_ptr()

        # Resize to smaller size (should reuse storage)
        x.resize_(50)
        self.assertEqual(x.numel(), 50)
        # Storage should still be available

        # Resize to original size
        x.resize_(100)
        self.assertEqual(x.numel(), 100)

    def test_empty_cache_operation(self):
        """Test empty cache functionality."""
        # Allocate some tensors
        x = torch.empty(1000, device="mcpu")
        y = torch.empty(2000, device="mcpu")

        # Delete tensors
        del x, y
        gc.collect()

        # Note: OpenRegDeviceAllocator.emptyCache is currently a no-op
        # This test ensures it doesn't crash
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_memory_format_allocation(self):
        """Test allocation with different memory formats."""
        # Channels last format
        x = torch.empty(2, 3, 4, 4, device="mcpu", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "mcpu")
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

        # Contiguous format (default)
        y = torch.empty(
            2, 3, 4, 4, device="mcpu", memory_format=torch.contiguous_format
        )
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(y.is_contiguous())

    def test_large_allocation(self):
        """Test large memory allocation."""
        # Allocate a large tensor (10MB approximately)
        size = 10 * 1024 * 1024 // 4  # 10MB in float32
        x = torch.empty(size, device="mcpu")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.numel(), size)

    def test_sequential_allocations_and_deallocations(self):
        """Test sequential allocation and deallocation patterns."""
        for i in range(10):
            x = torch.empty(1000 + i * 100, device="mcpu")
            self.assertEqual(x.device.type, "mcpu")
            # Let tensor go out of scope
            del x
        gc.collect()

    def test_allocation_with_requires_grad(self):
        """Test allocation of tensors with gradient tracking."""
        x = torch.empty(100, device="mcpu", requires_grad=True)
        self.assertEqual(x.device.type, "mcpu")
        self.assertTrue(x.requires_grad)

        y = torch.randn(100, device="mcpu", requires_grad=True)
        self.assertEqual(y.device.type, "mcpu")
        self.assertTrue(y.requires_grad)

    def test_storage_operations(self):
        """Test storage-level operations."""
        x = torch.randn(100, device="mcpu")
        storage = x.storage()

        # Verify storage is on correct device
        self.assertTrue(storage.device.type == "mcpu")

        # Verify storage size
        self.assertGreaterEqual(storage.size(), x.numel())

    def test_tensor_from_blob(self):
        """Test creating tensors that reference existing memory."""
        x = torch.randn(100, device="mcpu")

        # Create a view that references the same data
        y = x.view_as(x)

        # They should share the same underlying storage
        self.assertEqual(x.data_ptr(), y.data_ptr())

        # Modifying one should affect the other
        x.fill_(5.0)
        self.assertTrue(torch.all(y == 5.0))


class TestMemoryLeaks(TestCase):
    """Test cases for detecting memory leaks in OpenRegDeviceAllocator."""

    def setUp(self):
        """Reset memory state before each test."""
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup

    def test_no_leak_simple_allocations(self):
        """Test that simple allocations don't leak memory."""
        # Warm-up
        for _ in range(10):
            x = torch.empty(1000, device="mcpu")
            del x
        gc.collect()
        time.sleep(0.1)

        # Perform many allocations and deallocations
        iterations = 1000
        for i in range(iterations):
            x = torch.empty(1000, device="mcpu")
            del x

            if i % 100 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

        # If there were leaks, this would have accumulated significant memory
        # The test passes if no exception/crash occurred

    def test_no_leak_varying_sizes(self):
        """Test that allocations of varying sizes don't leak."""
        iterations = 500
        sizes = [100, 500, 1000, 5000, 10000]

        for i in range(iterations):
            size = sizes[i % len(sizes)]
            x = torch.empty(size, device="mcpu")
            del x

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_copies(self):
        """Test that tensor copies don't leak memory."""
        iterations = 300

        for i in range(iterations):
            # Create tensor
            x = torch.randn(500, device="mcpu")

            # Copy to CPU
            cpu_copy = x.cpu()

            # Copy back to device
            device_copy = cpu_copy.to("mcpu")

            # Clone
            cloned = device_copy.clone()

            # Delete all
            del x, cpu_copy, device_copy, cloned

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_views(self):
        """Test that tensor views don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.randn(1000, device="mcpu")

            # Create various views
            view1 = x.view(10, 100)
            view2 = x[100:200]
            view3 = x.reshape(20, 50)

            # Delete views and original
            del view1, view2, view3, x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_inplace_operations(self):
        """Test that inplace operations don't leak memory."""
        iterations = 500

        for i in range(iterations):
            x = torch.ones(1000, device="mcpu")

            # Multiple inplace operations
            x.add_(1)
            x.mul_(2)
            x.div_(2)
            x.sub_(1)

            del x

            if i % 100 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_with_gradients(self):
        """Test that tensors with gradients don't leak."""
        iterations = 300

        for i in range(iterations):
            x = torch.randn(100, device="mcpu", requires_grad=True)
            y = torch.randn(100, device="mcpu", requires_grad=True)

            # Operation that creates computation graph
            z = x + y

            # Delete all
            del x, y, z

            if i % 50 == 0:
                gc.collect()

        gc.collect()
        time.sleep(0.1)

    def test_no_leak_repeated_large_allocations(self):
        """Test repeated large allocations for memory leaks."""
        # Large tensor size (50MB)
        size = 50 * 1024 * 1024 // 4
        iterations = 50

        for i in range(iterations):
            x = torch.empty(size, device="mcpu")
            del x
            gc.collect()
            time.sleep(0.05)  # Allow time for cleanup

        # Final cleanup
        gc.collect()
        time.sleep(0.1)

    def test_leak_detection_with_statistics(self):
        """Test memory leak detection using allocation patterns."""
        # This test verifies that after many alloc/dealloc cycles,
        # the allocator properly frees memory

        num_cycles = 10
        allocations_per_cycle = 100

        for cycle in range(num_cycles):
            tensors = []

            # Allocate many tensors
            for i in range(allocations_per_cycle):
                t = torch.empty(1000, device="mcpu")
                tensors.append(t)

            # Verify all allocated
            self.assertEqual(len(tensors), allocations_per_cycle)

            # Delete all
            tensors.clear()
            gc.collect()
            time.sleep(0.05)

        # Final verification - if there were leaks, memory would be exhausted
        # The test passes if we can still allocate
        final_tensor = torch.empty(10000, device="mcpu")
        self.assertEqual(final_tensor.device.type, "mcpu")
        del final_tensor


class TestPinMemory(TestCase):
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        tensor = torch.randn(10)
        self.assertFalse(tensor.is_pinned())
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        slice_tensor = pinned_tensor[2:5]
        self.assertTrue(slice_tensor.is_pinned())

        tensor = torch.randn(10)
        storage = tensor.storage()
        self.assertFalse(storage.is_pinned("mcpu"))
        pinned_storage = storage.pin_memory("mcpu")
        self.assertTrue(pinned_storage.is_pinned("mcpu"))

        tensor = torch.randn(10)
        untyped_storage = tensor.untyped_storage()
        self.assertFalse(untyped_storage.is_pinned("mcpu"))
        pinned_untyped_storage = untyped_storage.pin_memory("mcpu")
        self.assertTrue(pinned_untyped_storage.is_pinned("mcpu"))


class TestMultiDeviceAllocation(TestCase):
    """Test basic multi-device allocation functionality."""

    def setUp(self):
        self.device_count = torch.mcpu.device_count()
        if self.device_count < 2:
            self.skipTest("This test requires 2 OpenReg devices, but only 1 is available")
        gc.collect()

    def tearDown(self):
        """Restore device 0 to avoid affecting subsequent tests."""
        torch.mcpu.set_device(0)
        gc.collect()

    def test_allocation_on_device_1(self):
        torch.mcpu.set_device(1)
        x = torch.empty(100, device="mcpu:1")
        self.assertEqual(x.device.type, "mcpu")
        self.assertEqual(x.device.index, 1)

    def test_simultaneous_device_allocations(self):
        """Test allocations on both devices simultaneously."""
        x = torch.empty(100, device="mcpu:0")
        y = torch.empty(200, device="mcpu:1")

        self.assertEqual(x.device.index, 0)
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

    def test_memory_isolation_between_devices(self):
        """Test that memory allocations are isolated between devices."""

        tensors_dev0 = [torch.empty(1000, device="mcpu:0") for _ in range(10)]
        tensors_dev1 = [torch.empty(1000, device="mcpu:1") for _ in range(10)]

        # Verify all device 0 tensors are on device 0
        for t in tensors_dev0:
            self.assertEqual(t.device.index, 0)

        # Verify all device 1 tensors are on device 1
        for t in tensors_dev1:
            self.assertEqual(t.device.index, 1)

        # Pointers should be different
        ptrs_dev0 = {t.data_ptr() for t in tensors_dev0}
        ptrs_dev1 = {t.data_ptr() for t in tensors_dev1}
        self.assertEqual(
            len(ptrs_dev0 & ptrs_dev1), 0, "Devices should not share pointers"
        )

    def test_alternating_device_allocations(self):
        """Test alternating allocations between devices."""
        tensors = []
        for i in range(20):
            device_idx = i % 2
            t = torch.empty(100 + i, device=f"mcpu:{device_idx}")
            self.assertEqual(t.device.index, device_idx)
            tensors.append(t)

        # Verify all tensors retained correct device assignment
        for i, t in enumerate(tensors):
            expected_device = i % 2
            self.assertEqual(t.device.index, expected_device)


class TestCrossDeviceOperations(TestCase):
    """Test cross-device tensor operations."""

    def setUp(self):
        self.device_count = torch.mcpu.device_count()
        if self.device_count < 2:
            self.skipTest("This test requires 2 OpenReg devices, but only 1 is available")
        gc.collect()

    def tearDown(self):
        """Restore device 0 to avoid affecting subsequent tests."""
        torch.mcpu.set_device(0)
        gc.collect()

    def test_tensor_to_different_device(self):
        """Test moving tensor from one device to another."""
        # Create on device 0
        x = torch.randn(100, device="mcpu:0")
        self.assertEqual(x.device.index, 0)

        # Move to device 1
        y = x.to("mcpu:1")
        self.assertEqual(y.device.index, 1)
        self.assertNotEqual(x.data_ptr(), y.data_ptr())

        # Values should be the same
        self.assertTrue(torch.allclose(x.cpu(), y.cpu()))

    def test_bidirectional_device_transfer(self):
        """Test transferring tensor back and forth between devices."""
        original = torch.randn(100, device="mcpu:0")
        original_cpu = original.cpu()

        # 0 -> 1
        on_dev1 = original.to("mcpu:1")
        self.assertTrue(torch.allclose(original_cpu, on_dev1.cpu()))

        # 1 -> 0
        back_to_dev0 = on_dev1.to("mcpu:0")
        self.assertTrue(torch.allclose(original_cpu, back_to_dev0.cpu()))


class TestCachingDeviceAllocator(TestCase):
    """Tests for the caching device allocator (DeviceCachingAllocator)."""

    def setUp(self):
        gc.collect()
        x = torch.empty(1, device="mcpu")
        del x
        gc.collect()
        torch.mcpu.empty_cache()
        torch.mcpu.reset_accumulated_memory_stats(0)
        torch.mcpu.reset_peak_memory_stats(0)

    def tearDown(self):
        gc.collect()
        torch.mcpu.empty_cache()

    def test_empty_cache_releases_reserved_memory(self):
        """empty_cache() should return all unoccupied blocks to the OS."""
        x = torch.empty(1000, device="mcpu")
        del x
        gc.collect()

        # After deleting x, reserved > 0 but allocated == 0
        stats_before = torch.mcpu.memory_stats(0)
        self.assertGreater(stats_before["reserved_bytes.all.current"], 0)
        self.assertEqual(stats_before["allocated_bytes.all.current"], 0)

        torch.mcpu.empty_cache()
        stats_after = torch.mcpu.memory_stats(0)
        self.assertEqual(stats_after["reserved_bytes.all.current"], 0)

    @skipIfTorchDynamo()
    def test_accelerator_empty_cache_synchronizes_stream_work(self):
        """torch.accelerator.empty_cache() should wait for pending MCPU work."""
        stream = torch.Stream(device="mcpu:0")
        marker = torch.empty(1, dtype=torch.int64, device="mcpu")

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(marker, 17, 500)

        self.assertFalse(stream.query())
        torch.accelerator.empty_cache()
        self.assertTrue(stream.query())
        self.assertEqual(torch.ops.mcpu.first_element_int(marker), 17)

    @skipIfTorchDynamo()
    def test_empty_cache_does_not_deadlock_on_pending_tensor_release(self):
        """empty_cache must not hold allocator mutex while waiting for kernels."""
        env = os.environ.copy()
        env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import torch
                    import torch_mcpu  # noqa: F401

                    stream = torch.Stream(device="mcpu:0")
                    with stream:
                        torch.ops.mcpu.stream_sleep_fill_(
                            torch.empty(1024, dtype=torch.int64, device="mcpu"),
                            7,
                            500,
                        )
                    torch.accelerator.empty_cache()
                    """
                ),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_memory_stats_allocated_tracks_live_tensors(self):
        """allocated_bytes tracks bytes in active tensors."""
        torch.mcpu.reset_accumulated_memory_stats(0)
        stats0 = torch.mcpu.memory_stats(0)
        alloc0 = stats0["allocated_bytes.all.current"]

        x = torch.empty(512, dtype=torch.float32, device="mcpu")  # 2 KiB
        stats1 = torch.mcpu.memory_stats(0)
        alloc1 = stats1["allocated_bytes.all.current"]
        self.assertGreater(alloc1, alloc0)

        del x
        gc.collect()
        stats2 = torch.mcpu.memory_stats(0)
        alloc2 = stats2["allocated_bytes.all.current"]
        self.assertLess(alloc2, alloc1)

    def test_memory_allocated_tracks_live_tensors(self):
        """memory_allocated() should expose allocated_bytes current."""
        alloc0 = torch.mcpu.memory_allocated(0)

        x = torch.empty(512, dtype=torch.float32, device="mcpu")  # 2 KiB
        alloc1 = torch.mcpu.memory_allocated(torch.device("mcpu:0"))
        self.assertGreater(alloc1, alloc0)

        del x
        gc.collect()
        torch.mcpu.empty_cache()
        self.assertEqual(torch.mcpu.memory_allocated("mcpu:0"), alloc0)

    def test_caching_reuses_freed_blocks(self):
        """A freed block should be reused for the next allocation of similar size."""
        x = torch.empty(1000, device="mcpu")
        ptr1 = x.data_ptr()
        del x
        gc.collect()

        y = torch.empty(1000, device="mcpu")
        ptr2 = y.data_ptr()
        del y

        # The caching allocator must have reused the block
        self.assertEqual(ptr1, ptr2, "Caching allocator should reuse freed blocks")

    def test_record_stream_delays_reuse_for_pointer_only_task(self):
        """record_stream keeps raw-pointer stream work safe after Tensor release."""
        torch.mcpu.synchronize()
        torch.mcpu.empty_cache()
        gc.collect()

        default_stream = torch.mcpu.default_stream()
        torch.mcpu.set_stream(default_stream)
        stream = torch.Stream(device="mcpu:0")
        numel = 3_000_000
        marker = torch.empty(1, dtype=torch.int64, device="mcpu")
        data = torch.ones(numel, dtype=torch.float32, device="mcpu")
        begin = torch.empty(1, dtype=torch.int64, device="mcpu")
        end = torch.empty(1, dtype=torch.int64, device="mcpu")
        torch.mcpu.synchronize()
        data_ptr = data.data_ptr()
        stats_with_data = torch.mcpu.memory_stats(0)

        with stream:
            torch.ops.mcpu.stream_sleep_fill_(marker, 1, 300)
            torch.ops.mcpu.pointer_gap_for_loop(
                data,
                begin,
                end,
                0,
                numel,
                0,  # raw pointer-only launch mode
                0,  # c10::getTime timer
            )

        data.record_stream(stream)
        del data
        gc.collect()
        self.assertFalse(stream.query())
        stats_pending = torch.mcpu.memory_stats(0)
        self.assertLess(
            stats_pending["allocated_bytes.all.current"],
            stats_with_data["allocated_bytes.all.current"],
            "destroying the Tensor should release allocated_bytes accounting",
        )

        replacement = torch.empty(numel, dtype=torch.float32, device="mcpu")
        stats_replacement = torch.mcpu.memory_stats(0)
        self.assertGreater(
            stats_replacement["reserved_bytes.all.current"],
            stats_pending["reserved_bytes.all.current"],
            "pending cross-stream use should force a new block reservation",
        )
        self.assertNotEqual(
            replacement.data_ptr(),
            data_ptr,
            "record_stream should prevent reuse before the side stream finishes",
        )

        stream.synchronize()
        reserved_before_recovered = torch.mcpu.memory_stats(0)[
            "reserved_bytes.all.current"
        ]
        recovered = torch.empty(numel, dtype=torch.float32, device="mcpu")
        stats_recovered = torch.mcpu.memory_stats(0)
        self.assertEqual(
            stats_recovered["reserved_bytes.all.current"],
            reserved_before_recovered,
            "completed stream events should make the pending block reusable",
        )
        self.assertEqual(
            recovered.data_ptr(),
            data_ptr,
            "the recorded block should return to the reusable pool after stream completion",
        )
        self.assertAlmostEqual(
            recovered.cpu()[0].item(),
            1.0000011,
            places=5,
            msg="the delayed pointer-only task should have written before reuse",
        )

    def test_reset_peak_memory_stats(self):
        """reset_peak_memory_stats clears the peak counter."""
        x = torch.empty(4096, device="mcpu")
        stats = torch.mcpu.memory_stats(0)
        peak_before = stats["reserved_bytes.all.peak"]
        self.assertGreater(peak_before, 0)

        torch.mcpu.reset_peak_memory_stats(0)
        stats_reset = torch.mcpu.memory_stats(0)
        peak_after_reset = stats_reset["reserved_bytes.all.peak"]
        self.assertLessEqual(peak_after_reset, peak_before)
        del x

    def test_reset_accumulated_memory_stats(self):
        """reset_accumulated_memory_stats zeroes allocated/freed counters."""
        x = torch.empty(1000, device="mcpu")
        del x
        gc.collect()

        stats_before = torch.mcpu.memory_stats(0)
        self.assertGreater(stats_before.get("num_alloc_retries", 0) +
                           stats_before["allocated_bytes.all.allocated"] +
                           stats_before["allocated_bytes.all.freed"], 0)

        torch.mcpu.reset_accumulated_memory_stats(0)
        stats_after = torch.mcpu.memory_stats(0)
        self.assertEqual(stats_after["allocated_bytes.all.allocated"], 0)
        self.assertEqual(stats_after["allocated_bytes.all.freed"], 0)

    def test_large_and_small_blocks_classified_separately(self):
        """Allocations ≤1 MB go to the small pool; allocations >1 MB go to large."""
        # Small: 256 KiB
        small = torch.empty(256 * 1024 // 4, dtype=torch.float32, device="mcpu")
        # Large: 2 MiB
        large = torch.empty(2 * 1024 * 1024 // 4, dtype=torch.float32, device="mcpu")

        self.assertEqual(small.device.type, "mcpu")
        self.assertEqual(large.device.type, "mcpu")
        del small, large

    def test_multiple_alloc_free_cycles_no_growth(self):
        """Repeated alloc/free cycles should not grow reserved memory indefinitely."""
        # Warm-up pass: establish the cached pool
        for _ in range(20):
            t = torch.empty(1024, device="mcpu")
            del t
        gc.collect()

        stats_before = torch.mcpu.memory_stats(0)
        reserved_before = stats_before["reserved_bytes.all.current"]

        # Another 100 cycles — reserved should stay flat (blocks reused)
        for _ in range(100):
            t = torch.empty(1024, device="mcpu")
            del t

        stats_after = torch.mcpu.memory_stats(0)
        reserved_after = stats_after["reserved_bytes.all.current"]
        self.assertEqual(reserved_before, reserved_after,
                         "Reserved memory should not grow when blocks are reused")


class TestCachingHostAllocator(TestCase):
    """Tests for the caching host (pinned) allocator (CachingHostAllocatorImpl)."""

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_unpin_basic(self):
        """Pinning a tensor returns a pinned tensor; deleting it should not crash."""
        t = torch.randn(100)
        self.assertFalse(t.is_pinned())
        pinned = t.pin_memory()
        self.assertTrue(pinned.is_pinned())
        del pinned

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_data_matches_original(self):
        """Pinned copy must have the same values as the source."""
        t = torch.arange(50, dtype=torch.float32)
        pinned = t.pin_memory()
        self.assertTrue(torch.equal(t, pinned))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_to_device_async(self):
        """Non-blocking transfer from pinned memory to mcpu device should work."""
        t = torch.randn(64)
        pinned = t.pin_memory()
        device_tensor = pinned.to("mcpu", non_blocking=True)
        self.assertEqual(device_tensor.device.type, "mcpu")
        self.assertTrue(torch.allclose(t, device_tensor.cpu()))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pinned_memory_reuse(self):
        """Freed pinned blocks should be reused for subsequent pin_memory calls."""
        t = torch.randn(1000)
        p1 = t.pin_memory()
        ptr1 = p1.data_ptr()
        del p1
        gc.collect()

        p2 = t.pin_memory()
        ptr2 = p2.data_ptr()
        del p2
        # The caching host allocator should reuse the freed pinned block
        self.assertEqual(ptr1, ptr2,
                         "Caching host allocator should reuse freed pinned blocks")

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_large_tensor(self):
        """Large pinned allocations should succeed."""
        t = torch.randn(10 * 1024 * 1024 // 4)  # 10 MiB
        pinned = t.pin_memory()
        self.assertTrue(pinned.is_pinned())
        del pinned


class TestSharedXmalloc(TestCase):
    """Test shared backing through the installed MCPU allocation API."""

    def run_child(self, code, path, size="4096", **extra_env):
        env = os.environ.copy()
        env.pop("TORCH_MCPU_SHARED_BASE", None)
        env.update(TORCH_MCPU_SHARED_SIM_FILE=path, TORCH_MCPU_SHARED_SIZE=size)
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.update(extra_env)
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            env=env, check=False, capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_premapped_backend_with_supplied_mapped_region(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required for pre-mapped backend smoke test")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "xmalloc-premapped-smoke")
            result = subprocess.run(
                [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                 "-DTORCH_MCPU_SHARED_USE_PREMAPPED=1", f"-I{root / 'include'}",
                 f"-I{root / 'csrc'}", str(root / "csrc/runtime/OpenRegSharedAllocator.cpp"),
                 str(root / "tests/xmalloc_premapped_smoke.cpp"), "-o", binary],
                check=False, capture_output=True, text=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [binary], check=False, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_alignment_capacity_and_tensor_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_child("""
                import gc
                import torch
                from torch_mcpu.shared_memory import xmalloc

                a = xmalloc(1)
                b = xmalloc(257)
                assert a.data_ptr() % 256 == 0
                assert b.data_ptr() - a.data_ptr() == 256
                assert a.device.type == 'mcpu'
                assert b.dtype == torch.uint8 and b.numel() == 257
                assert not b.untyped_storage().resizable()
                b.fill_(19)
                view = b[1:]
                del b
                gc.collect()
                torch.testing.assert_close(view.cpu(), torch.full((256,), 19, dtype=torch.uint8))
                try:
                    xmalloc(257)  # only 256 aligned bytes remain
                except RuntimeError as error:
                    assert 'exhausted' in str(error)
                else:
                    raise AssertionError('expected exhaustion')
                c = xmalloc(256)  # failed allocation must not advance the cursor
                assert c.data_ptr() - a.data_ptr() == 768
                bf16 = c.view(torch.bfloat16).view(4, 32)
                bf16.fill_(7)
                torch.testing.assert_close(bf16.cpu(), torch.full((4, 32), 7, dtype=torch.bfloat16))
                torch.testing.assert_close(view.cpu(), torch.full((256,), 19, dtype=torch.uint8))
                # Ordinary device allocation still works independently.
                ordinary = torch.ones(8, device='mcpu')
                torch.testing.assert_close(ordinary.cpu(), torch.ones(8))
            """, os.path.join(directory, "arena"), size="1024")

    def test_two_rank_updates_and_group_isolation(self):
        # Two TP groups use different files concurrently. Each pair exchanges
        # readiness bytes only; all tensor data travels through xmalloc backing.
        code = textwrap.dedent("""
            import os
            import socket
            import sys
            import time
            import torch
            from torch_mcpu.shared_memory import xmalloc

            rank, fd, seed, attach_late = map(int, sys.argv[1:])
            channel = socket.socket(fileno=fd)
            channel.settimeout(20)
            if rank == 1 and attach_late:
                assert channel.recv(1) == b'I'
            cache = xmalloc(4096).view(torch.int32).view(4, 256)
            extra = xmalloc(257)
            assert extra.data_ptr() - cache.data_ptr() == 4096
            assert cache.data_ptr() % 256 == 0
            if rank == 0:
                cache.fill_(seed)
                torch.mcpu.synchronize()
                channel.sendall(b'I')
                assert channel.recv(1) == b'A'
            else:
                if not attach_late:
                    assert channel.recv(1) == b'I'
                torch.testing.assert_close(cache.cpu(), torch.full((4, 256), seed, dtype=torch.int32))
                channel.sendall(b'A')
            expected = torch.full((4, 256), seed, dtype=torch.int32)
            for step in range(32):
                block = step % 4
                values = torch.arange(256, dtype=torch.int32) + seed + step * 256
                expected[block].copy_(values)
                if rank == 0:
                    cache[block].copy_(values)
                    # A host notification alone does not finish queued kernels.
                    torch.mcpu.synchronize()
                    channel.sendall(b'R')
                    assert channel.recv(1) == b'A'
                else:
                    assert channel.recv(1) == b'R'
                    time.sleep(0.002)
                    torch.testing.assert_close(cache.cpu(), expected)
                    # The writer must not reuse the block until this read ends.
                    channel.sendall(b'A')
            channel.close()
        """)
        processes = []
        with tempfile.TemporaryDirectory() as directory:
            try:
                for group in range(2):
                    env = os.environ.copy()
                    env.pop("TORCH_MCPU_SHARED_BASE", None)
                    env.update(
                        TORCH_MCPU_SHARED_SIM_FILE=os.path.join(directory, f"group-{group}"),
                        TORCH_MCPU_SHARED_SIZE="8192",
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                    )
                    left, right = socket.socketpair()
                    try:
                        for rank, channel in enumerate((left, right)):
                            processes.append(subprocess.Popen(
                                [sys.executable, "-c", code, str(rank),
                                 str(channel.fileno()), str(1000 * group), str(group)],
                                env=env, pass_fds=(channel.fileno(),),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True,
                            ))
                    finally:
                        left.close()
                        right.close()
                failures = []
                for process in processes:
                    stdout, stderr = process.communicate(timeout=45)
                    if process.returncode != 0:
                        failures.append(f"PID {process.pid}: {stdout}{stderr}")
                self.assertFalse(failures, "\n".join(failures))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=10)

    def test_invalid_configuration_and_sizes_do_not_allocate(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_child("""
                import os
                import pytest
                from torch_mcpu.shared_memory import xmalloc

                path = os.environ['TORCH_MCPU_SHARED_SIM_FILE']
                for value in ('', '-1', '1x', '18446744073709551616', '0', '255'):
                    os.environ['TORCH_MCPU_SHARED_SIZE'] = value
                    with pytest.raises((RuntimeError, ValueError)):
                        xmalloc(1)
                del os.environ['TORCH_MCPU_SHARED_SIZE']
                with pytest.raises((RuntimeError, ValueError), match='TORCH_MCPU_SHARED_SIZE'):
                    xmalloc(1)
                os.environ['TORCH_MCPU_SHARED_SIZE'] = '0x1000'
                for size in (0, -1, 1 << 64):
                    with pytest.raises((RuntimeError, ValueError, OverflowError)):
                        xmalloc(size)
                assert not os.path.exists(path)
                del os.environ['TORCH_MCPU_SHARED_SIM_FILE']
                with pytest.raises((RuntimeError, ValueError), match='TORCH_MCPU_SHARED_SIM_FILE'):
                    xmalloc(1)
                os.environ['TORCH_MCPU_SHARED_SIM_FILE'] = path
                assert xmalloc(4096).data_ptr() % 256 == 0
            """, os.path.join(directory, "arena"))

    def test_existing_region_size_mismatch_preserves_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "arena")
            with open(path, "wb") as region:
                region.write(b'x' * 512)
            self.run_child("""
                from torch_mcpu.shared_memory import xmalloc
                try:
                    xmalloc(256)
                except (RuntimeError, ValueError) as error:
                    assert 'size differs' in str(error)
                else:
                    raise AssertionError('size mismatch accepted')
            """, path)
            with open(path, "rb") as region:
                self.assertEqual(region.read(), b'x' * 512)

    def test_fixed_base_never_overwrites_existing_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_child("""
                import ctypes
                import mmap
                import os
                from torch_mcpu.shared_memory import xmalloc

                occupied = mmap.mmap(-1, mmap.PAGESIZE)
                address = ctypes.addressof(ctypes.c_char.from_buffer(occupied))
                occupied[:4] = b'keep'
                os.environ['TORCH_MCPU_SHARED_BASE'] = hex(address)
                try:
                    xmalloc(1)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('overwrote an existing mapping')
                assert occupied[:4] == b'keep'
                occupied.close()
                # This single-threaded setup has now made the address available.
                allocation = xmalloc(256)
                assert allocation.data_ptr() == address
                allocation.fill_(23)
                assert allocation.cpu().tolist() == [23] * 256
            """, os.path.join(directory, "arena"), size=str(os.sysconf("SC_PAGESIZE")))


if __name__ == "__main__":
    run_tests()
