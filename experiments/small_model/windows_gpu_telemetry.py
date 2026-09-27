"""Read-only Windows GPU telemetry with explicit CUDA/WDDM identity and missing-data rejection."""
import ctypes
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time

MIB = 2 ** 20
MEMORY_LIMIT = 300 * MIB
INSTANCE = re.compile(r"^(?:pid_(?P<pid>\d+)_)?luid_(?P<high>0x[0-9a-f]{8})_(?P<low>0x[0-9a-f]{8})_phys_(?P<node>\d+)$", re.I)
COUNTER_PATH = re.compile(
    r"^\\\\[^\\]+\\(?P<object>gpu (?:process|adapter) memory)\((?P<instance>[^()\\#]+)"
    r"(?:#(?P<index>\d+))?\)\\(?P<counter>dedicated usage|shared usage)$", re.I)
COUNTER_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$names = @{}
Get-Process | ForEach-Object { $names[$_.Id] = $_.ProcessName }
$query = '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
         '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage'
$samples = @((Get-Counter -Counter $query -ErrorAction Stop).CounterSamples | ForEach-Object {
    $processId = if ($_.InstanceName -match '^pid_(\d+)_') { [int]$Matches[1] } else { 0 }
    [pscustomobject]@{path=$_.Path; instance=$_.InstanceName; object=$_.Path.Split('\')[-2].Split('(')[0];
        counter=$_.Path.Split('\')[-1]; bytes=$_.CookedValue; status=$_.Status; name=$names[$processId]}
})
[pscustomobject]@{captured_utc=[DateTime]::UtcNow.ToString('o'); samples=$samples} | ConvertTo-Json -Depth 4 -Compress
"""


def _call(library, name, types, *arguments):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = ctypes.c_int
    code = function(*arguments)
    if code:
        raise RuntimeError(f"{name} returned error {code}; sensor reading unavailable")


def cuda_identity(ordinal=0):
    if os.name != "nt" or type(ordinal) is not int or ordinal < 0:
        raise ValueError("Require a Windows CUDA device ordinal")
    library = ctypes.WinDLL(str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvcuda.dll"))
    _call(library, "cuInit", [ctypes.c_uint], 0)
    device = ctypes.c_int()
    _call(library, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), ordinal)
    luid = (ctypes.c_char * 8)()
    node = ctypes.c_uint()
    _call(library, "cuDeviceGetLuid", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int],
          luid, ctypes.byref(node), device)
    name, pci = ctypes.create_string_buffer(256), ctypes.create_string_buffer(32)
    for function, output in (("cuDeviceGetName", name), ("cuDeviceGetPCIBusId", pci)):
        _call(library, function, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], output, len(output), device)
    context = ctypes.c_void_p()
    _call(library, "cuCtxGetCurrent", [ctypes.POINTER(ctypes.c_void_p)], ctypes.byref(context))
    low, high = struct.unpack("<II", bytes(luid))
    if not node.value or node.value & (node.value - 1):
        raise ValueError("Require exactly one physical GPU node for unambiguous memory attribution")
    return {"cuda_ordinal": ordinal, "name": name.value.decode("ascii"), "pci_bus_id": pci.value.decode("ascii"),
            "luid": f"0x{high:08x}_0x{low:08x}", "node_mask": node.value, "physical_node": node.value.bit_length() - 1,
            "current_cuda_context_present": bool(context.value)}


class _Memory(ctypes.Structure):
    _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]


class _Utilization(ctypes.Structure):
    _fields_ = [(key, ctypes.c_uint) for key in ("gpu", "memory")]


class NvidiaTelemetry:
    """NVML reads only; no GPU setting, context allocation, tensor or workload is created."""

    def __init__(self, ordinal=0):
        self.identity = cuda_identity(ordinal)
        candidates = [Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvml.dll",
                      Path(os.environ.get("ProgramW6432", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvml.dll"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("The installed NVIDIA monitoring library is unavailable")
        self.library = ctypes.WinDLL(str(path))
        self.handle = ctypes.c_void_p()
        self.initialized = False
        try:
            _call(self.library, "nvmlInit_v2", [])
            self.initialized = True
            _call(self.library, "nvmlDeviceGetHandleByPciBusId_v2", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
                  self.identity["pci_bus_id"].encode("ascii"), ctypes.byref(self.handle))
            for function, key in (("nvmlDeviceGetName", "nvml_name"), ("nvmlDeviceGetUUID", "uuid")):
                value = ctypes.create_string_buffer(256)
                _call(self.library, function, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint], self.handle, value, len(value))
                self.identity[key] = value.value.decode("ascii")
            if self.identity["name"] != self.identity["nvml_name"]:
                raise ValueError("CUDA and NVML refer to different GPU names")
            self.event_function = ("nvmlDeviceGetCurrentClocksEventReasons" if hasattr(self.library, "nvmlDeviceGetCurrentClocksEventReasons")
                                   else "nvmlDeviceGetCurrentClocksThrottleReasons")
        except Exception:
            self.close()
            raise

    def sample(self):
        if not self.initialized:
            raise RuntimeError("Monitoring library is closed")
        temp, clock = ctypes.c_uint(), ctypes.c_uint()
        reasons, memory, utilization = ctypes.c_ulonglong(), _Memory(), _Utilization()
        _call(self.library, "nvmlDeviceGetTemperature", [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
              self.handle, 0, ctypes.byref(temp))
        _call(self.library, "nvmlDeviceGetClockInfo", [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
              self.handle, 1, ctypes.byref(clock))
        _call(self.library, self.event_function, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)], self.handle, ctypes.byref(reasons))
        _call(self.library, "nvmlDeviceGetMemoryInfo", [ctypes.c_void_p, ctypes.POINTER(_Memory)], self.handle, ctypes.byref(memory))
        _call(self.library, "nvmlDeviceGetUtilizationRates", [ctypes.c_void_p, ctypes.POINTER(_Utilization)], self.handle, ctypes.byref(utilization))
        return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(), "temp_c": temp.value,
                "sm_mhz": clock.value, "util_pct": utilization.gpu, "memory_util_pct": utilization.memory,
                "total_bytes": memory.total, "free_bytes": memory.free, "used_bytes": memory.used,
                "clocks_event_reasons": reasons.value, "clocks_event_api": self.event_function,
                "sw_thermal": bool(reasons.value & 0x20), "hw_thermal": bool(reasons.value & 0x40)}

    def close(self):
        if self.initialized:
            _call(self.library, "nvmlShutdown", [])
            self.initialized = False


def read_windows_counters():
    if os.name != "nt":
        raise ValueError("Windows performance counters are required")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", COUNTER_SCRIPT],
                            capture_output=True, text=True, timeout=30, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or not isinstance(value.get("samples"), list) or not value.get("captured_utc"):
        raise ValueError("Missing Windows counter sample metadata")
    return value


def selected_memory(reading, identity, own_pid):
    """Keep adapter/node attribution and apply the existing per-process limit, never a sum of PIDs."""
    if (not isinstance(reading, dict) or not isinstance(reading.get("samples"), list) or not isinstance(identity, dict)
            or type(own_pid) is not int or own_pid <= 0
            or not re.fullmatch(r"0x[0-9a-f]{8}_0x[0-9a-f]{8}", identity.get("luid", ""), re.I)
            or type(identity.get("physical_node")) is not int or identity["physical_node"] < 0):
        raise ValueError("Missing counter, process or adapter identity")
    seen, adapter, processes = set(), {}, {}
    ignored_adapters = set()
    for row in reading["samples"]:
        if (not isinstance(row, dict) or not {"path", "instance", "bytes", "counter", "object", "status"}.issubset(row)
                or any(not isinstance(row[k], str) for k in ("path", "instance", "counter", "object"))):
            raise ValueError("Incomplete Windows counter row")
        match = INSTANCE.fullmatch(row["instance"])
        path = COUNTER_PATH.fullmatch(row["path"])
        value = row["bytes"]
        kind, object_name = row["counter"].lower(), row["object"].lower()
        if (not match or not path or kind not in ("dedicated usage", "shared usage")
                or object_name not in ("gpu process memory", "gpu adapter memory")
                or type(row["status"]) is not int or row["status"] != 0
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or int(value) != value):
            raise ValueError("Invalid, unavailable or unknown Windows memory counter")
        if (path["instance"].lower() != row["instance"].lower() or path["object"].lower() != object_name
                or path["counter"].lower() != kind):
            raise ValueError("Counter path and sample identity disagree")
        key = (object_name, row["instance"].lower(), int(path["index"] or 0), kind)
        if key in seen:
            raise ValueError("Duplicate counter instance")
        seen.add(key)
        luid = f"{match['high']}_{match['low']}".lower()
        node = int(match["node"])
        pid = int(match["pid"]) if match["pid"] else None
        if (pid is not None) != (object_name == "gpu process memory"):
            raise ValueError("Counter object and instance disagree")
        if luid != identity["luid"].lower() or node != identity["physical_node"]:
            ignored_adapters.add((luid, node))
            continue
        if int(path["index"] or 0) != 0:
            raise ValueError("Multiple indexed instances on the target adapter have ambiguous memory accounting")
        destination = adapter if pid is None else processes.setdefault(pid, {"pid": pid, "name": row.get("name")})
        destination[kind.split()[0] + "_bytes"] = int(value)
    required = {"dedicated_bytes", "shared_bytes"}
    if not required.issubset(adapter) or any(not required.issubset(p) for p in processes.values()):
        raise ValueError("Missing target adapter or process counters; unavailable is not clean")
    ordered = sorted(processes.values(), key=lambda p: p["pid"])
    foreign = [p for p in ordered if p["pid"] != own_pid and p["dedicated_bytes"] > MEMORY_LIMIT]
    return {"status": "complete", "luid": identity["luid"].lower(), "physical_node": identity["physical_node"],
            "own_pid": own_pid, "adapter": adapter, "processes": ordered, "foreign_over_limit": foreign,
            "contaminated": bool(foreign), "per_process_limit_bytes": MEMORY_LIMIT,
            "other_adapter_nodes": [list(v) for v in sorted(ignored_adapters)],
            "scope": "Per-process memory may count allocations shared across processes. Do not sum these rows into physical GPU memory, and do not infer active compute contention from allocated bytes alone."}


def thermal_eligible(sample, maximum_c=70):
    required = {"temp_c", "sw_thermal", "hw_thermal"}
    if (not isinstance(sample, dict) or not required.issubset(sample)
            or type(sample["temp_c"]) is not int or not 0 <= sample["temp_c"] <= 130
            or type(sample["sw_thermal"]) is not bool or type(sample["hw_thermal"]) is not bool):
        raise ValueError("Missing or invalid thermal telemetry")
    return sample["temp_c"] <= maximum_c and not sample["sw_thermal"] and not sample["hw_thermal"]
