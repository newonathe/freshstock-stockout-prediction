"""Resource limits and cohort routing. No model score is used anywhere in this decision."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from freshstock import data as b3
from freshstock.expanded import resources

GIB = 2 ** 30


class LimitTests(unittest.TestCase):
    def test_ceiling_is_the_smaller_rule(self):
        self.assertEqual(resources.memory_ceiling(20 * GIB), int(.8 * 20 * GIB))
        self.assertEqual(resources.memory_ceiling(6 * GIB), 4 * GIB)
        self.assertEqual(resources.memory_ceiling(10 * GIB), 8 * GIB)
        self.assertEqual(resources.memory_ceiling(2 * GIB), 0)
        self.assertEqual(resources.memory_ceiling(1 * GIB), 0)

    def test_snapshot_has_real_readings(self):
        s = resources.snapshot()
        self.assertGreater(s["total_ram"], s["available_ram"])
        self.assertGreater(s["available_ram"], 0)
        self.assertGreater(s["working_set"], 1_000_000)
        self.assertGreater(s["disk_free"], 0)
        self.assertGreaterEqual(s["logical_cpus"], s["affinity_cpus"])
        json.dumps(s, allow_nan=False)


class RoutingTests(unittest.TestCase):
    LIMITS = {"available_ram": 8 * GIB, "disk_free": 20 * GIB, "seconds_left": 10_000}

    def tiers(self, **overrides):
        base = {"peak_bytes": 1 * GIB, "seconds": 100, "disk_bytes": GIB}
        return {t: {**base, **overrides.get(f"{t[1]}_{t[0]}", {})} for t in resources.TIERS}

    def test_order_prefers_coverage(self):
        self.assertEqual(resources.TIERS, [(50000, "full"), (50000, "reduced"), (25000, "full"), (25000, "reduced"),
                                           (10000, "full"), (10000, "reduced")])
        self.assertEqual(resources.choose_tier(self.tiers(), self.LIMITS)["tier"], [50000, "full"])

    def test_each_limit_moves_to_the_next_tier(self):
        for bad in ({"peak_bytes": 7 * GIB}, {"seconds": 10_001}, {"disk_bytes": 16 * GIB}):
            got = resources.choose_tier(self.tiers(full_50000=bad), self.LIMITS)
            self.assertEqual(got["tier"], [50000, "reduced"])
            self.assertTrue(got["rejected"][0]["reason"])

    def test_missing_projection_is_not_safe(self):
        tiers = self.tiers()
        del tiers[(50000, "full")]
        self.assertEqual(resources.choose_tier(tiers, self.LIMITS)["tier"], [50000, "reduced"])

    def test_nothing_safe_is_refused(self):
        for limit in ({"available_ram": 2 * GIB}, {"disk_free": 5 * GIB}, {"seconds_left": 50}):
            with self.assertRaises(b3.DataError):
                resources.choose_tier(self.tiers(), {**self.LIMITS, **limit})

    def test_scores_are_not_accepted_as_input(self):
        tiers = self.tiers()
        tiers[(50000, "full")]["average_precision"] = .7
        with self.assertRaises(b3.DataError):
            resources.choose_tier(tiers, self.LIMITS)

    def test_fewest_threads_within_five_percent_of_fastest(self):
        self.assertEqual(resources.choose_threads({1: 80, 2: 45, 4: 26, 8: 25, 16: 25.5}), 4)
        self.assertEqual(resources.choose_threads({1: 10, 2: 10.1}), 1)
        self.assertEqual(resources.choose_threads({1: 10, 8: 9}), 8)


class GuardTests(unittest.TestCase):
    def test_records_peak_and_lowest_free_memory(self):
        with resources.Guard(interval=.01) as guard:
            block = bytearray(50_000_000)
            time.sleep(.1)
        report = guard.report()
        self.assertGreater(report["peak_additional_bytes"], 30_000_000)
        self.assertGreater(report["seconds"], .09)
        self.assertGreater(report["min_available_ram"], 0)
        self.assertEqual(report["stopped"], None)
        del block

    def test_three_low_samples_stop_the_owned_process(self):
        stopped, readings = [], iter([3 * GIB, GIB, GIB, 3 * GIB, GIB, GIB, GIB] + [3 * GIB] * 100)
        def read():
            return {"available_ram": next(readings), "working_set": 1, "disk_free": 50 * GIB}
        with resources.Guard(interval=.005, read=read, stop=stopped.append, persist=0) as guard:
            time.sleep(.3)
        self.assertEqual(len(stopped), 1)
        self.assertIn("memory", stopped[0])
        self.assertEqual(guard.report()["stopped"], stopped[0])

    def test_short_dip_is_not_persistent(self):
        """Fast sampling must not turn a dip of a fraction of a second into a stop; the rule is about five seconds."""
        def low():
            return {"available_ram": GIB, "working_set": 1, "disk_free": 50 * GIB}
        stopped = []
        with resources.Guard(interval=.005, read=low, stop=stopped.append):
            time.sleep(.2)
        self.assertEqual(stopped, [])
        with resources.Guard(interval=.005, read=low, stop=stopped.append, persist=.1):
            time.sleep(.4)
        self.assertEqual(len(stopped), 1)

    def test_low_disk_and_deadline_stop(self):
        def read():
            return {"available_ram": 9 * GIB, "working_set": 1, "disk_free": GIB}
        stopped = []
        with resources.Guard(interval=.005, read=read, stop=stopped.append):
            time.sleep(.1)
        self.assertIn("disk", stopped[0])
        stopped = []
        with resources.Guard(interval=.005, timeout=.02, stop=stopped.append):
            time.sleep(.2)
        self.assertIn("time", stopped[0])

    def test_default_stop_leaves_a_diagnostic_and_exits(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "stop.json"
            code = ("import sys,time;from freshstock.expanded import resources as r;"
                    "g=r.Guard(interval=.01,timeout=.02,stop=r.exit_with_diagnostic(sys.argv[1]));"
                    "g.__enter__();time.sleep(20)")
            done = subprocess.run([sys.executable, "-c", code, str(log)], cwd=b3.ROOT, timeout=15)
            self.assertEqual(done.returncode, 1)
            self.assertIn("time", json.loads(log.read_text())["reason"])


if __name__ == "__main__":
    unittest.main()
