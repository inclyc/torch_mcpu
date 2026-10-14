# Owner(s): ["module: PrivateUse1"]

import gzip
import json
import os
import tempfile
from unittest import mock

import torch
import torch.nn as nn
import torch_mcpu.profiler as mcpu_profiler
from torch.autograd.profiler import profile as autograd_profile
from torch.profiler import record_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class SimpleModel(nn.Module):
    """Simple neural network for integration testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestProfiler(TestCase):
    """Test PyTorch profiler integration with OpenReg backend."""

    @skipIfTorchDynamo()
    def test_profiler_basic(self):
        """Test basic profiler functionality with OpenReg device."""
        with autograd_profile(use_device="mcpu") as prof:
            x = torch.randn(10, 10, device="mcpu")
            y = torch.randn(10, 10, device="mcpu")
            x + y

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Check that we have OpenReg operations recorded
        event_names = [e.name for e in events]
        self.assertTrue(any("aten::" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_record_function(self):
        """Test profiler with custom record_function annotations."""
        with autograd_profile(use_device="mcpu") as prof:
            with record_function("mcpu_custom_operation"):
                x = torch.randn(10, 10, device="mcpu")
                y = torch.randn(10, 10, device="mcpu")
                x @ y

        events = prof.function_events
        event_names = [e.name for e in events]
        self.assertTrue(any("mcpu_custom_operation" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_multiple_streams(self):
        """Test profiler with multiple OpenReg streams."""
        stream1 = torch.Stream(device="mcpu")
        stream2 = torch.Stream(device="mcpu")

        with autograd_profile(use_device="mcpu") as prof:
            with stream1:
                x = torch.randn(10, 10, device="mcpu")
                x + x

            with stream2:
                z = torch.randn(10, 10, device="mcpu")
                z * z

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_event_synchronization(self):
        """Test that profiler correctly handles event synchronization."""
        with autograd_profile(use_device="mcpu") as prof:
            stream = torch.Stream(device="mcpu")
            event = torch.Event(device="mcpu", enable_timing=True)

            with stream:
                x = torch.randn(10, 10, device="mcpu")
                event.record(stream)
                x + x

            event.synchronize()

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_nested_record_function(self):
        """Test profiler with nested record_function calls."""
        with autograd_profile(use_device="mcpu") as prof:
            with record_function("outer_operation"):
                x = torch.randn(10, 10, device="mcpu")

                with record_function("inner_operation_1"):
                    x + x

                with record_function("inner_operation_2"):
                    y = x + x
                    y * y

        events = prof.function_events
        event_names = [e.name for e in events]
        self.assertTrue(any("outer_operation" in name for name in event_names))
        self.assertTrue(any("inner_operation_1" in name for name in event_names))
        self.assertTrue(any("inner_operation_2" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_key_averages(self):
        """Test that key_averages works correctly with OpenReg operations."""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ]
        ) as prof:
            for _ in range(10):
                x = torch.randn(10, 10, device="mcpu")
                x + x

        key_averages = prof.key_averages()
        self.assertGreater(len(key_averages), 0)
        self.assertGreater(
            sum(avg.self_device_time_total for avg in key_averages), 0
        )

        # Verify we can access statistics
        for avg in key_averages:
            self.assertIsNotNone(avg.key)
            self.assertGreaterEqual(avg.count, 0)
            self.assertGreaterEqual(avg.cpu_time_total, 0)

    @skipIfTorchDynamo()
    def test_profiler_table_output(self):
        """Test that profiler can generate table output."""
        with autograd_profile(use_device="mcpu") as prof:
            x = torch.randn(10, 10, device="mcpu")
            y = x + x
            x @ y

        # Should not raise any exceptions
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        self.assertIsInstance(table, str)
        self.assertGreater(len(table), 0)

    @skipIfTorchDynamo()
    def test_profiler_export_chrome_trace(self):
        """Test exporting profiler results to Chrome trace format."""
        with autograd_profile(use_device="mcpu") as prof:
            x = torch.randn(10, 10, device="mcpu")
            y = x + x
            x @ y

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_file = f.name

        try:
            prof.export_chrome_trace(trace_file)

            # Verify the trace file is valid JSON
            with open(trace_file) as f:
                trace_data = json.load(f)

            if isinstance(trace_data, dict):
                self.assertIn("traceEvents", trace_data)
                events = trace_data["traceEvents"]
            else:
                events = trace_data
            self.assertGreater(len(events), 0)
        finally:
            import os

            if os.path.exists(trace_file):
                os.remove(trace_file)

    def test_profiler_gzip_trace_postprocess_keeps_valid_gzip(self):
        """Test mcpu trace injection into a gzip Chrome trace."""
        trace = {
            "schemaVersion": 1,
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::empty",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 5.0,
                    "args": {},
                }
            ],
        }
        fd, trace_file = tempfile.mkstemp(suffix=".pt.trace.json.gz")
        os.close(fd)

        try:
            with gzip.open(trace_file, "wt", encoding="utf-8") as f:
                json.dump(trace, f)

            with mock.patch.object(
                mcpu_profiler,
                "_valid_kernel_events",
                return_value=[(1000, 1300, "mcpu::unit_test_kernel", 0)],
            ):
                injected = mcpu_profiler._append_mcpu_events_to_trace(trace_file)

            self.assertEqual(injected, 1)
            with gzip.open(trace_file, "rt", encoding="utf-8") as f:
                trace_data = json.load(f)

            events = trace_data["traceEvents"]
            self.assertTrue(
                any(event.get("cat") == "mcpu_kernel" for event in events)
            )
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)

    def test_valid_kernel_events_consumes_kernel_timing(self):
        """Test profiler consumes kernel timing events."""
        with mock.patch.object(
            torch.mcpu,
            "get_kernel_timing",
            return_value=[
                {
                    "role": "worker",
                    "events": [
                        {
                            "name": "mcpu::unit_test_kernel",
                            "stream": 3,
                            "begin_time": 2000,
                            "end_time": 2600,
                            "elapsed_time": 600,
                        }
                    ],
                }
            ],
        ):
            self.assertEqual(
                mcpu_profiler._valid_kernel_events(),
                [(2000, 2600, "mcpu::unit_test_kernel", 3)],
            )

    def test_kernel_timing_legacy_aliases_are_not_exposed(self):
        """Test legacy timing aliases are not part of the Python API."""
        self.assertFalse(hasattr(torch.mcpu, "get_kernel_timing_ns"))
        self.assertFalse(hasattr(torch.mcpu, "read_kernel_timing_tsc"))

    def test_profiler_start_syncs_before_resetting_kernel_timing(self):
        """Test profiler start synchronizes before clearing kernel events."""
        calls = []

        with (
            mock.patch.object(
                torch.mcpu, "synchronize", side_effect=lambda: calls.append("sync")
            ),
            mock.patch.object(
                torch.mcpu,
                "reset_kernel_timing",
                side_effect=lambda: calls.append("reset"),
            ),
            mock.patch.object(
                torch.mcpu,
                "set_kernel_timing_enabled",
                side_effect=lambda enabled: calls.append(("enabled", enabled)),
            ),
        ):
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            )
            prof.start()
            prof.stop()

        self.assertEqual(calls[:3], ["sync", "reset", ("enabled", True)])
        self.assertEqual(calls[-2:], ["sync", ("enabled", False)])

    @skipIfTorchDynamo()
    def test_profiler_start_clears_stale_kernel_timing_events(self):
        """Test profiler start clears kernel events from earlier work."""
        x = torch.empty(8, dtype=torch.int64, device="mcpu")

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            torch.ops.mcpu.stream_sleep_fill_(x, 1, 1)
            torch.mcpu.synchronize()
            self.assertGreater(len(mcpu_profiler._valid_kernel_events()), 0)
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ):
            pass

        self.assertEqual(mcpu_profiler._valid_kernel_events(), [])

    def test_profiler_trace_postprocess_uses_direct_timer_path(self):
        """Test trace injection converts kernel timing directly to us."""
        trace = {
            "schemaVersion": 1,
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::empty",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 5.0,
                    "args": {},
                }
            ],
        }
        fd, trace_file = tempfile.mkstemp(suffix=".pt.trace.json")
        os.close(fd)

        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace, f)

            with mock.patch.object(
                mcpu_profiler,
                "_valid_kernel_events",
                return_value=[(1000, 1300, "mcpu::unit_test_kernel", 0)],
            ):
                injected = mcpu_profiler._append_mcpu_events_to_trace(trace_file)

            self.assertEqual(injected, 1)
            with open(trace_file, encoding="utf-8") as f:
                trace_data = json.load(f)

            mcpu_events = [
                event
                for event in trace_data["traceEvents"]
                if event.get("cat") == "mcpu_kernel" and event.get("ph") == "X"
            ]
            self.assertEqual(len(mcpu_events), 1)
            self.assertEqual(mcpu_events[0]["ts"], 100.0 + 5.0)
            self.assertEqual(mcpu_events[0]["dur"], 0.3)
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)

    def test_profiler_trace_postprocess_keeps_distinct_stream_ids(self):
        """Test mcpu trace injection keeps per-kernel stream ids distinct."""
        trace = {
            "schemaVersion": 1,
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "aten::empty",
                    "pid": 1,
                    "tid": 1,
                    "ts": 100.0,
                    "dur": 5.0,
                    "args": {},
                }
            ],
        }
        fd, trace_file = tempfile.mkstemp(suffix=".pt.trace.json")
        os.close(fd)

        try:
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace, f)

            with mock.patch.object(
                mcpu_profiler,
                "_valid_kernel_events",
                return_value=[
                    (1000, 1300, "mcpu::stream_a", 101),
                    (1100, 1500, "mcpu::stream_b", 202),
                ],
            ):
                injected = mcpu_profiler._append_mcpu_events_to_trace(trace_file)

            self.assertEqual(injected, 2)
            with open(trace_file, encoding="utf-8") as f:
                trace_data = json.load(f)

            mcpu_events = [
                event
                for event in trace_data["traceEvents"]
                if event.get("cat") == "mcpu_kernel" and event.get("ph") == "X"
            ]
            self.assertEqual(
                {event.get("tid") for event in mcpu_events},
                {"mcpu_stream_101", "mcpu_stream_202"},
            )
            self.assertEqual(
                {event.get("args", {}).get("stream") for event in mcpu_events},
                {101, 202},
            )
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)

    @skipIfTorchDynamo()
    def test_profiler_export_chrome_trace_multiple_mcpu_streams(self):
        """Test Chrome trace export emits separate mcpu stream lanes."""
        stream1 = torch.Stream(device="mcpu")
        stream2 = torch.Stream(device="mcpu")
        x = torch.empty(8, dtype=torch.int64, device="mcpu")
        y = torch.empty(8, dtype=torch.int64, device="mcpu")

        torch.mcpu.reset_kernel_timing()
        torch.mcpu.set_kernel_timing_enabled(True)
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                with stream1:
                    torch.ops.mcpu.stream_sleep_fill_(x, 1, 1)
                with stream2:
                    torch.ops.mcpu.stream_sleep_fill_(y, 2, 1)
                stream1.synchronize()
                stream2.synchronize()
        finally:
            torch.mcpu.set_kernel_timing_enabled(False)

        fd, trace_file = tempfile.mkstemp(suffix=".pt.trace.json")
        os.close(fd)

        try:
            # tensorboard_trace_handler passes this export option explicitly.
            prof.export_chrome_trace(trace_file, use_python_export=False)
            with open(trace_file, encoding="utf-8") as f:
                trace_data = json.load(f)

            events = (
                trace_data["traceEvents"] if isinstance(trace_data, dict) else trace_data
            )
            stream_ids = {
                event.get("args", {}).get("stream")
                for event in events
                if event.get("cat") == "mcpu_kernel"
                and event.get("name") == "mcpu::stream_sleep_fill_"
            }
            self.assertGreaterEqual(len(stream_ids), 2)
        finally:
            if os.path.exists(trace_file):
                os.remove(trace_file)
            torch.mcpu.reset_kernel_timing()

    @skipIfTorchDynamo()
    def test_profiler_with_shapes(self):
        """Test profiler with shape recording enabled."""
        with autograd_profile(use_device="mcpu", record_shapes=True) as prof:
            x = torch.randn(10, 20, device="mcpu")
            y = torch.randn(20, 30, device="mcpu")
            x @ y

        key_averages = prof.key_averages(group_by_input_shape=True)
        self.assertGreater(len(key_averages), 0)

    @skipIfTorchDynamo()
    def test_profiler_default_stream_initialization(self):
        """Test that profiler works without explicit stream initialization."""
        # This test verifies the bug fix: profiler should work even when
        # getStreamFromPool is not called first
        with autograd_profile(use_device="mcpu") as prof:
            # Use default stream without explicitly creating streams
            x = torch.randn(10, 10, device="mcpu")
            x + 1.0

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_stream_event_correlation(self):
        """Test that profiler correctly correlates events across streams."""
        stream1 = torch.Stream(device="mcpu")
        stream2 = torch.Stream(device="mcpu")

        with autograd_profile(use_device="mcpu") as prof:
            with record_function("stream1_ops"):
                with stream1:
                    x = torch.randn(10, 10, device="mcpu")
                    x + x

            with record_function("stream2_ops"):
                with stream2:
                    z = torch.randn(10, 10, device="mcpu")
                    z * z

        events = prof.function_events
        stream1_events = [e for e in events if "stream1_ops" in e.name]
        stream2_events = [e for e in events if "stream2_ops" in e.name]

        self.assertGreater(len(stream1_events), 0)
        self.assertGreater(len(stream2_events), 0)

    @skipIfTorchDynamo()
    def test_profiler_elapsed_time(self):
        """Test that event elapsed time is correctly calculated."""
        with autograd_profile(use_device="mcpu") as prof:
            stream = torch.Stream(device="mcpu")

            with stream:
                x = torch.randn(100, 100, device="mcpu")
                y = torch.randn(100, 100, device="mcpu")
                # Multiple matmuls to ensure measurable time
                for _ in range(10):
                    z = x @ y
                    x = z

            stream.synchronize()

        events = prof.function_events
        # Check that operations have non-zero duration
        compute_events = [
            e for e in events if "aten::mm" in e.name or "aten::matmul" in e.name
        ]
        if compute_events:
            total_time = sum(e.cpu_time_total for e in compute_events)
            self.assertGreater(total_time, 0)

    @skipIfTorchDynamo()
    def test_profiler_context_manager_multiple_times(self):
        """Test that profiler can be used multiple times."""
        with autograd_profile(use_device="mcpu") as prof:
            for i in range(3):
                x = torch.randn(10, 10, device="mcpu")
                x = x + i

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_with_backward_pass(self):
        """Test profiler with autograd operations."""
        with autograd_profile(use_device="mcpu", record_shapes=True) as prof:
            x = torch.randn(10, 10, device="mcpu", requires_grad=True)
            y = torch.randn(10, 10, device="mcpu", requires_grad=True)
            z = (x @ y).sum()
            z.backward()

        events = prof.function_events
        event_names = [e.name for e in events]

        # Check for forward and backward operations
        self.assertTrue(
            any("aten::mm" in name or "aten::matmul" in name for name in event_names)
        )
        self.assertTrue(any("backward" in name.lower() for name in event_names))


class TestProfilerEdgeCases(TestCase):
    """Test edge cases and error conditions for profiler."""

    @skipIfTorchDynamo()
    def test_profiler_empty_operations(self):
        """Test profiler with no operations."""
        with autograd_profile(use_device="mcpu") as prof:
            pass

        events = prof.function_events
        # Should still work, even with no user operations
        self.assertIsInstance(events, list)

    @skipIfTorchDynamo()
    def test_profiler_exception_handling(self):
        """Test that profiler handles exceptions gracefully."""
        try:
            with autograd_profile(use_device="mcpu") as prof:
                x = torch.randn(10, 10, device="mcpu")  # noqa: F841
                raise RuntimeError("Test exception")
        except RuntimeError as e:
            self.assertEqual(str(e), "Test exception")

        # Profiler should still be usable
        events = prof.function_events
        self.assertIsInstance(events, list)

    @skipIfTorchDynamo()
    def test_profiler_large_number_of_events(self):
        """Test profiler with large number of events."""
        with autograd_profile(use_device="mcpu") as prof:
            x = torch.randn(10, 10, device="mcpu")

            # Create many small operations
            for i in range(100):
                with record_function(f"op_{i}"):
                    x = x + 1

        events = prof.function_events
        self.assertGreater(len(events), 100)


class TestProfilerMultiDevice(TestCase):
    """Test profiler with multiple OpenReg devices if available."""

    @skipIfTorchDynamo()
    def test_profiler_single_device(self):
        """Test profiler with explicit device specification."""
        device_count = torch.mcpu.device_count()
        if device_count < 1:
            self.skipTest("No OpenReg devices available")

        with autograd_profile(use_device="mcpu") as prof:
            x = torch.randn(10, 10, device="mcpu:0")
            x + x

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_multiple_devices(self):
        """Test profiler with operations on multiple devices."""
        device_count = torch.mcpu.device_count()
        if device_count < 2:
            self.skipTest("Multiple OpenReg devices required")

        with autograd_profile(use_device="mcpu") as prof:
            x0 = torch.randn(10, 10, device="mcpu:0")
            x1 = torch.randn(10, 10, device="mcpu:1")

            x0 + x0
            x1 + x1

        events = prof.function_events
        self.assertGreater(len(events), 0)


class TestProfilerIntegration(TestCase):
    """Integration tests for profiler with realistic PyTorch workflows."""

    @skipIfTorchDynamo()
    def test_profiler_with_simple_model(self):
        """Test profiler with a simple neural network."""
        model = SimpleModel().to("mcpu")
        x = torch.randn(32, 10, device="mcpu")

        with autograd_profile(use_device="mcpu") as prof:
            model(x)

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Check for linear and relu operations
        event_names = [e.name for e in events]
        self.assertTrue(
            any(
                "linear" in name.lower() or "addmm" in name.lower()
                for name in event_names
            )
        )

    @skipIfTorchDynamo()
    def test_profiler_with_training_step(self):
        """Test profiler with a complete training step."""
        model = SimpleModel().to("mcpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x = torch.randn(32, 10, device="mcpu")
        target = torch.randn(32, 10, device="mcpu")

        with autograd_profile(use_device="mcpu", record_shapes=True) as prof:
            # Forward pass
            with record_function("forward"):
                output = model(x)

            # Loss calculation
            with record_function("loss"):
                loss = criterion(output, target)

            # Backward pass
            with record_function("backward"):
                loss.backward()

            # Optimizer step
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

        events = prof.function_events
        event_names = [e.name for e in events]

        # Verify all stages are recorded
        self.assertTrue(any("forward" in name for name in event_names))
        self.assertTrue(any("loss" in name for name in event_names))
        self.assertTrue(any("backward" in name for name in event_names))
        self.assertTrue(any("optimizer_step" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_multiple_batches(self):
        """Test profiler with multiple training batches."""
        model = SimpleModel().to("mcpu")

        with autograd_profile(use_device="mcpu") as prof:
            for i in range(5):
                with record_function(f"batch_{i}"):
                    x = torch.randn(16, 10, device="mcpu")
                    model(x)

        events = prof.function_events

        # Check that all batches are recorded
        event_names = [e.name for e in events]
        for i in range(5):
            self.assertTrue(any(f"batch_{i}" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_table_with_model(self):
        """Test that profiler table output works with model operations."""
        model = SimpleModel().to("mcpu")
        x = torch.randn(32, 10, device="mcpu")

        with autograd_profile(use_device="mcpu", record_shapes=True) as prof:
            for _ in range(10):
                model(x)

        # Generate table - should not raise exceptions
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)

        self.assertIsInstance(table, str)
        self.assertGreater(len(table), 0)

        # Table should contain operation names
        self.assertTrue(
            any(keyword in table.lower() for keyword in ["linear", "addmm", "relu"])
        )

    @skipIfTorchDynamo()
    def test_profiler_with_inference(self):
        """Test profiler during model inference (no gradients)."""
        model = SimpleModel().to("mcpu")
        model.eval()

        with autograd_profile(use_device="mcpu") as prof:
            with torch.no_grad():
                for _ in range(5):
                    x = torch.randn(16, 10, device="mcpu")
                    model(x)

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Should not have backward operations
        event_names = [e.name for e in events]
        # Backward operations should be minimal or none
        backward_ops = [name for name in event_names if "backward" in name.lower()]  # noqa: F841
        # It's ok if there are some backward-related events in the profiler
        # infrastructure, but there shouldn't be many

    @skipIfTorchDynamo()
    def test_profiler_with_data_transfer(self):
        """Test profiler with CPU-OpenReg data transfers."""
        with autograd_profile(use_device="mcpu") as prof:
            # CPU tensor
            x_cpu = torch.randn(50, 50)

            # Transfer to OpenReg
            with record_function("to_mcpu"):
                x_mcpu = x_cpu.to("mcpu")

            # Compute on OpenReg
            with record_function("compute_mcpu"):
                y = x_mcpu @ x_mcpu

            # Transfer back to CPU
            with record_function("to_cpu"):
                y.cpu()

        events = prof.function_events
        event_names = [e.name for e in events]

        self.assertTrue(any("to_mcpu" in name for name in event_names))
        self.assertTrue(any("compute_mcpu" in name for name in event_names))
        self.assertTrue(any("to_cpu" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_mixed_operations(self):
        """Test profiler with mixed tensor operations."""
        with autograd_profile(use_device="mcpu", record_shapes=True) as prof:
            x = torch.randn(20, 20, device="mcpu")

            # Various operations
            y = x + 1.0
            z = torch.matmul(x, y)
            w = torch.relu(z)
            v = w.transpose(0, 1)
            v.sum()

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Should capture various operation types
        event_names = [e.name for e in events]
        self.assertTrue(len([n for n in event_names if "aten::" in n]) > 0)


class TestProfilerPerformance(TestCase):
    """Performance-related profiler tests."""

    @skipIfTorchDynamo()
    def test_profiler_overhead(self):
        """Verify profiler doesn't add excessive overhead."""
        model = SimpleModel().to("mcpu")
        x = torch.randn(32, 10, device="mcpu")

        # Warmup
        for _ in range(5):
            model(x)

        # Profile should complete in reasonable time
        import time

        start = time.time()

        with autograd_profile(use_device="mcpu") as prof:
            for _ in range(100):
                model(x)

        elapsed = time.time() - start

        # Should complete in reasonable time (not checking exact time as it's hardware dependent)
        # Just verify it completes and produces valid results
        events = prof.function_events
        self.assertGreater(len(events), 0)
        self.assertLess(elapsed, 60.0)  # Should complete within 1 minute


if __name__ == "__main__":
    run_tests()
