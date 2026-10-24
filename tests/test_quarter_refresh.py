"""Checks against Q2 filing anchors and malformed quarterly inputs."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from model import ModelParams, dscr_analysis, rate_shock_curve, run_simulation
from validate import ConfigValidationError, load_and_validate

ROOT = Path(__file__).resolve().parents[1]


class QuarterRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_and_validate(ROOT / "config/2026Q2.yaml", verbose=False)

    def test_reported_book_value_and_rate_anchors(self):
        # Q2 10-Q GAAP/Economic BV reconciliation and Item 3 Shock Table.
        self.assertEqual(round(self.cfg["_derived"]["gaap_bvps_baseline"], 2), 12.71)
        self.assertEqual(round(self.cfg["_derived"]["econ_bvps_baseline"], 2), 13.20)
        curve = rate_shock_curve(self.cfg, [-100, -50, 0, 50, 100])
        np.testing.assert_allclose(curve["dnpv_mm"],
                                   [85.270, 54.928, 0, -79.514, -183.614], atol=1e-9)

    def test_liquidity_and_stress_paths(self):
        for mode in ("negative_equity", "ramp"):
            with self.subTest(mode=mode):
                params = ModelParams.from_config(self.cfg, n_sims=1000, default_mode=mode)
                res = run_simulation(self.cfg, params)
                self.assertAlmostEqual(res.available_liquidity, 391.178, places=3)
                self.assertEqual(len(res.econ_bvps), 1000)
                self.assertTrue(np.isfinite(res.econ_bvps).all())
                self.assertLess(np.median(res.econ_bvps), 13.20)

    def test_dscr_uses_disclosed_multifamily_aging(self):
        res = dscr_analysis(self.cfg, measure="dpd_60plus")
        mf = next(r for r in res["rows"] if r["key"] == "mf_transitional")
        self.assertAlmostEqual(mf["delinquency"], 88080 / 360373)
        self.assertEqual(mf["net_yield"], .0712)

    def test_inconsistent_aging_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        aging = cfg["delinquency_aging"]["segments"]["mf_transitional"]
        # Preserve the row total while corrupting the disclosed 60+ rate.
        aging["current"] += 10000
        aging["dpd_90plus"] -= 10000
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ConfigValidationError, "60\\+ DPD aging"):
                    load_and_validate(path, verbose=False)

    def test_prior_quarter_still_validates(self):
        load_and_validate(ROOT / "config/2026Q1.yaml", verbose=False)


if __name__ == "__main__":
    unittest.main()
