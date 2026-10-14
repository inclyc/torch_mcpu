import gzip
import json
import os
import tempfile

import torch


def _prefer_mcpu_profiler_fallback():
    try:
        import torch.autograd.profiler as autograd_profiler

        privateuse1 = torch.profiler.ProfilerActivity.PrivateUse1
        original_supported = torch.autograd._supported_activities

        def supported_without_mcpu_kineto():
            activities = set(original_supported())
            if torch._C._get_privateuse1_backend_name() == "mcpu":
                activities.discard(privateuse1)
            return activities

        torch.autograd._supported_activities = supported_without_mcpu_kineto
        autograd_profiler._supported_activities = supported_without_mcpu_kineto

        # Newer PyTorch selects native PrivateUse1 tracing from use_kineto
        # alone. MCPU supplies timing stubs, so explicitly select fallback.
        original_init = autograd_profiler.profile.__init__
        if not getattr(original_init, "_torch_mcpu_patched", False):

            def init_with_mcpu_fallback(self, *args, **kwargs):
                if kwargs.get("use_device") == "mcpu":
                    kwargs["use_kineto"] = False
                original_init(self, *args, **kwargs)

            init_with_mcpu_fallback._torch_mcpu_patched = True
            autograd_profiler.profile.__init__ = init_with_mcpu_fallback
    except Exception:
        pass


def _valid_kernel_events():
    events = []
    for thread_index, thread in enumerate(torch.mcpu.get_kernel_timing()):
        for event in thread.get("events", []):
            begin = int(event.get("begin_time", 0))
            end = int(event.get("end_time", 0))
            if begin == 0 or end <= begin:
                continue
            events.append(
                (
                    begin,
                    end,
                    str(event.get("name", "mcpu::kernel")),
                    int(
                        event.get(
                            "stream", thread.get("thread_index", thread_index)
                        )
                    ),
                )
            )
    events.sort(key=lambda item: item[0])
    return events


def _number_after(text, key, begin, end):
    pos = text.find(key, begin, end)
    if pos < 0:
        return None
    pos += len(key)
    while pos < min(end, len(text)) and text[pos] in " \t\r\n":
        pos += 1
    start = pos
    while pos < min(end, len(text)) and text[pos] in "0123456789+-.eE":
        pos += 1
    if pos == start:
        return None
    try:
        return float(text[start:pos])
    except ValueError:
        return None


def _first_mcpu_host_op_end_us(path):
    text = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return 0.0
            text = (text + chunk)[-(2 << 20) :]
            marker = '"cat": "cpu_op"'
            pos = text.find(marker)
            while pos >= 0:
                name_pos = text.find('"name": "', pos, pos + 4096)
                if name_pos >= 0:
                    name_start = name_pos + len('"name": "')
                    name_end = text.find('"', name_start)
                    if name_end >= 0:
                        name = text[name_start:name_end]
                        if name.startswith(("aten::", "mcpu::")):
                            ts = _number_after(text, '"ts":', pos, pos + 4096)
                            dur = _number_after(text, '"dur":', pos, pos + 4096)
                            if ts is not None:
                                return ts + (dur or 0.0)
                pos = text.find(marker, pos + len(marker))


def _find_trace_events_close(path):
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        offset = size
        tail = b""
        while offset > 0:
            read_size = min(1 << 20, offset)
            offset -= read_size
            handle.seek(offset)
            tail = handle.read(read_size) + tail
            pos = tail.rfind(b"]")
            if pos >= 0:
                return offset + pos
            tail = tail[:256]
    return -1


def _trace_events_need_comma(path, close_pos):
    with open(path, "rb") as handle:
        pos = close_pos - 1
        while pos >= 0:
            handle.seek(pos)
            byte = handle.read(1)
            if byte not in b" \t\r\n":
                return byte != b"["
            pos -= 1
    return False


def _append_mcpu_events(path):
    events = _valid_kernel_events()
    if not events:
        return 0

    close_pos = _find_trace_events_close(path)
    if close_pos < 0:
        return 0

    first_begin = events[0][0]
    anchor_us = _first_mcpu_host_op_end_us(path)
    need_comma = _trace_events_need_comma(path, close_pos)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, temp_path = tempfile.mkstemp(prefix=".mcpu_trace.", suffix=".json", dir=directory)
    injected = 0
    try:
        with os.fdopen(fd, "wb") as out, open(path, "rb") as src:
            remaining = close_pos
            while remaining > 0:
                chunk = src.read(min(1 << 20, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)

            if need_comma:
                out.write(b",")
            out.write(
                b'\n{"ph":"M","cat":"mcpu_kernel","name":"process_name",'
                b'"pid":"MCPU","args":{"name":"MCPU"}}'
            )
            for begin, end, name, stream_id in events:
                ts = anchor_us + (begin - first_begin) / 1000.0
                dur = (end - begin) / 1000.0
                out.write(b",\n")
                out.write(
                    json.dumps(
                        {
                            "ph": "X",
                            "cat": "mcpu_kernel",
                            "name": name,
                            "pid": "MCPU",
                            "tid": f"mcpu_stream_{stream_id}",
                            "ts": ts,
                            "dur": max(0.0, dur),
                            "args": {"stream": stream_id},
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                injected += 1
            out.write(b"\n")

            src.seek(close_pos)
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return injected


def _append_mcpu_events_to_trace(path):
    if not path.endswith(".gz"):
        return _append_mcpu_events(path)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    json_fd, json_path = tempfile.mkstemp(
        prefix=".mcpu_trace.", suffix=".json", dir=directory
    )
    gz_path = None
    try:
        with os.fdopen(json_fd, "wb") as out, gzip.open(path, "rb") as src:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)

        injected = _append_mcpu_events(json_path)

        gz_fd, gz_path = tempfile.mkstemp(
            prefix=".mcpu_trace.", suffix=".json.gz", dir=directory
        )
        with os.fdopen(gz_fd, "wb") as raw_out:
            with open(json_path, "rb") as src, gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_out
            ) as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        os.replace(gz_path, path)
        return injected
    except Exception:
        if gz_path is not None:
            try:
                os.unlink(gz_path)
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


def _patch_export_chrome_trace():
    try:
        profile_cls = torch.profiler.profile
        original_export = profile_cls.export_chrome_trace
    except Exception:
        return

    if getattr(original_export, "_torch_mcpu_patched", False):
        return

    def export_chrome_trace_with_mcpu(self, path, *args, **kwargs):
        result = original_export(self, path, *args, **kwargs)
        if (
            torch._C._get_privateuse1_backend_name() == "mcpu"
            and os.environ.get("TORCH_MCPU_PATCH_PROFILER_TRACE", "1") != "0"
        ):
            try:
                _append_mcpu_events_to_trace(path)
            except Exception:
                pass
        return result

    export_chrome_trace_with_mcpu._torch_mcpu_patched = True
    profile_cls.export_chrome_trace = export_chrome_trace_with_mcpu


def _profiler_kernel_timing_enabled():
    return (
        torch._C._get_privateuse1_backend_name() == "mcpu"
        and os.environ.get("TORCH_MCPU_PATCH_PROFILER_TRACE", "1") != "0"
    )


def _start_mcpu_kernel_timing_for_profiler():
    if not _profiler_kernel_timing_enabled():
        return
    torch.mcpu.synchronize()
    torch.mcpu.reset_kernel_timing()
    torch.mcpu.set_kernel_timing_enabled(True)


def _stop_mcpu_kernel_timing_for_profiler():
    if not _profiler_kernel_timing_enabled():
        return
    torch.mcpu.synchronize()
    torch.mcpu.set_kernel_timing_enabled(False)


def _patch_torch_profiler_start_stop():
    try:
        profile_cls = torch.profiler.profile
        original_start = profile_cls.start
        original_stop = profile_cls.stop
    except Exception:
        return

    if getattr(original_start, "_torch_mcpu_patched", False):
        return

    def start_with_mcpu_kernel_timing(self):
        _start_mcpu_kernel_timing_for_profiler()
        try:
            return original_start(self)
        except Exception:
            _stop_mcpu_kernel_timing_for_profiler()
            raise

    def stop_with_mcpu_kernel_timing(self):
        try:
            return original_stop(self)
        finally:
            _stop_mcpu_kernel_timing_for_profiler()

    start_with_mcpu_kernel_timing._torch_mcpu_patched = True
    stop_with_mcpu_kernel_timing._torch_mcpu_patched = True
    profile_cls.start = start_with_mcpu_kernel_timing
    profile_cls.stop = stop_with_mcpu_kernel_timing


def _patch_autograd_profiler_start_stop():
    try:
        profile_cls = torch.autograd.profiler.profile
        original_enter = profile_cls.__enter__
        original_exit = profile_cls.__exit__
    except Exception:
        return

    if getattr(original_enter, "_torch_mcpu_patched", False):
        return

    def enter_with_mcpu_kernel_timing(self):
        if getattr(self, "use_device", None) == "mcpu":
            _start_mcpu_kernel_timing_for_profiler()
        return original_enter(self)

    def exit_with_mcpu_kernel_timing(self, exc_type, exc_val, exc_tb):
        try:
            return original_exit(self, exc_type, exc_val, exc_tb)
        finally:
            if getattr(self, "use_device", None) == "mcpu":
                _stop_mcpu_kernel_timing_for_profiler()

    enter_with_mcpu_kernel_timing._torch_mcpu_patched = True
    exit_with_mcpu_kernel_timing._torch_mcpu_patched = True
    profile_cls.__enter__ = enter_with_mcpu_kernel_timing
    profile_cls.__exit__ = exit_with_mcpu_kernel_timing


def install():
    _prefer_mcpu_profiler_fallback()
    _patch_torch_profiler_start_stop()
    _patch_autograd_profiler_start_stop()
    _patch_export_chrome_trace()
