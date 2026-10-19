"""Statistics for paired evaluations on repeated objects across training seeds."""

import json
import unittest
import uuid

import numpy as np

from pims_net.evaluation.report_multiseed import build_report, paired_delta
from pims_net.paths import OUTPUT_ROOT


class MultiseedReportTests(unittest.TestCase):
    def test_pairing_averages_seeds_before_bootstrap(self):
        result = paired_delta([[1, 3], [-1, -3]])
        self.assertEqual(result["per_seed_delta"], [2, -2])
        self.assertEqual(result["paired_object_bootstrap_95ci"], [0, 0])
        self.assertEqual(result["delta"], 0)
        with self.assertRaises(ValueError):
            paired_delta([[np.nan]])

    def test_complete_table_and_direction_of_damage_improvement(self):
        root = OUTPUT_ROOT / f"report_test_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        files, folders = [], []
        try:
            for seed in (1, 2, 3):
                parent = root / f"seed_{seed}"
                folder = parent / "evaluation"
                folder.mkdir(parents=True)
                folders.extend([parent, folder])
                records = []
                for method in ("a0", "a1", "a2", "a3"):
                    for index in range(2):
                        records.append({"id": str(index), "method": method, "IoU@1": .5, "IoU@3": .6,
                                        "IoU@5": .8 if method == "a3" else .7, "NoC@90": 20,
                                        "BoundaryIoU@5": .4, "reached_90": False,
                                        "correct_region_damage_rate@2:5": .01 if method == "a3" else .02})
                path = folder / "per_object.jsonl"
                path.write_text("\n".join(json.dumps(item) for item in records))
                files.append(path)
            details = build_report(root, [1, 2, 3])
            files.extend(root / name for name in ("comparison.csv", "per_seed_comparison.csv", "stability.json", "comparison.md"))
            comparison = details["comparisons"]["a3_minus_a2"]
            self.assertEqual(comparison["IoU@5"]["improved_seed_count"], 3)
            self.assertEqual(comparison["correct_region_damage_rate@2:5"]["improved_seed_count"], 3)
            self.assertLess(comparison["correct_region_damage_rate@2:5"]["delta"], 0)
            self.assertEqual(len((root / "comparison.csv").read_text().splitlines()), 5)
        finally:
            for path in files:
                if path.exists():
                    path.unlink()
            for path in reversed(folders):
                path.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
