"""Measured laptop limits, the memory/time guard and the score-free cohort decision."""
import ctypes
import json
import os
import shutil
import threading
import time
from ctypes import wintypes
from pathlib import Path

from ..data import ROOT, require

GIB = 2 ** 30
RAM_FLOOR, DISK_FLOOR = 2 * GIB, 5 * GIB
TIERS = [(n, grid) for n in (50_000, 25_000, 10_000) for grid in ("full", "reduced")]
PROJECTION = {"peak_bytes", "seconds", "disk_bytes"}


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (n, ctypes.c_uint64) for n in ("total", "available", "commit_limit", "commit_available", "virtual_total",
                                       "virtual_available", "extended")]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ("peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged",
                                       "nonpaged", "pagefile", "peak_pagefile", "private")]


class _Power(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("battery", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                ("status", ctypes.c_ubyte), ("life", wintypes.DWORD), ("full", wintypes.DWORD)]


def snapshot(folder=ROOT):
    """Current Windows readings in bytes; no estimate and no sensor that the system does not expose."""
    kernel, memory, process, power = ctypes.windll.kernel32, _MemoryStatus(), _ProcessMemory(), _Power()
    memory.length, process.size = ctypes.sizeof(memory), ctypes.sizeof(process)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemory), wintypes.DWORD]
    require(kernel.GlobalMemoryStatusEx(ctypes.byref(memory)) and kernel.GetSystemPowerStatus(ctypes.byref(power)) and
            kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.size),
            "Windows did not return memory readings")
    return {"total_ram": memory.total, "available_ram": memory.available, "commit_available": memory.commit_available,
            "working_set": process.working_set, "peak_working_set": process.peak_working_set,
            "private_bytes": process.private, "disk_free": shutil.disk_usage(folder).free,
            "logical_cpus": os.cpu_count(), "affinity_cpus": os.process_cpu_count(), "on_ac_power": power.ac == 1}


def memory_ceiling(available):
    """Largest additional working set: 80% of available RAM, and always 2 GiB left for the system."""
    return max(0, min(int(.8 * available), available - RAM_FLOOR))


def choose_threads(median_seconds):
    """Fewest threads within 5% of the fastest median."""
    best = min(median_seconds.values())
    return min(n for n, s in median_seconds.items() if s <= best * 1.05)


def choose_tier(projections, limits):
    """First safe tier in the fixed coverage-first order; only time, memory and disk may be used."""
    rejected = []
    for tier in TIERS:
        p = projections.get(tier)
        if p is None:
            rejected.append({"tier": list(tier), "reason": "not projected"})
            continue
        require(set(p) == PROJECTION, "a projection may contain only time, memory and disk")
        reasons = [text for bad, text in (
            (p["peak_bytes"] > memory_ceiling(limits["available_ram"]), "memory ceiling"),
            (p["seconds"] > limits["seconds_left"], "time budget"),
            (limits["disk_free"] - p["disk_bytes"] < DISK_FLOOR, "disk floor")) if bad]
        if not reasons:
            return {"tier": list(tier), "rejected": rejected, "projection": p, "limits": limits}
        rejected.append({"tier": list(tier), "reason": ", ".join(reasons)})
    require(False, f"no cohort is safe on this laptop now: {rejected}")


def exit_with_diagnostic(path):
    def stop(reason):
        Path(path).write_text(json.dumps({"status": "STOPPED", "reason": reason}), encoding="utf-8")
        os._exit(1)
    return stop


class Guard:
    """Samples this process and the system; stops the owned work after three or more low readings over `persist` s."""

    def __init__(self, interval=1.5, timeout=None, read=snapshot, stop=None, persist=4.5):
        self.interval, self.timeout, self.read, self.stop, self.persist = interval, timeout, read, stop, persist
        self.stopped, self._done = None, threading.Event()

    def __enter__(self):
        first = self.read()
        self.start, self.base = time.perf_counter(), first["working_set"]
        self.peak, self.low = first["working_set"], first["available_ram"]
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._done.set()
        self._thread.join()
        self.seconds = time.perf_counter() - self.start
        last = self.read()
        self.peak, self.low = max(self.peak, last["working_set"]), min(self.low, last["available_ram"])

    def _watch(self):
        low, since = 0, 0.0
        while not self._done.wait(self.interval):
            s = self.read()
            self.peak, self.low = max(self.peak, s["working_set"]), min(self.low, s["available_ram"])
            low = low + 1 if s["available_ram"] < RAM_FLOOR else 0
            since = time.perf_counter() if low == 1 else since
            if low >= 3 and time.perf_counter() - since >= self.persist:
                return self._stop("free memory stayed below 2 GiB")
            if s["disk_free"] < DISK_FLOOR:
                return self._stop("free disk below 5 GiB")
            if self.timeout and time.perf_counter() - self.start > self.timeout:
                return self._stop("time limit reached")

    def _stop(self, reason):
        self.stopped = reason
        if self.stop:
            self.stop(reason)

    def report(self):
        return {"seconds": round(self.seconds, 3), "peak_additional_bytes": self.peak - self.base,
                "peak_working_set": self.peak, "min_available_ram": self.low, "stopped": self.stopped}
