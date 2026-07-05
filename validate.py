"""
validate.py — cross-foots the MFA stress config on load and HARD-FAILS with a
clear message if anything does not tie. Both model.py and app.py call
`load_and_validate()` at startup so a bad config can never silently produce a
wrong book value.

Checks performed
----------------
1. Financing columns (assets / non-MTM / MTM / securitized / senior / net equity)
   sum across segments to the reported totals, within a rounding tolerance that
   scales with the number of segments (so genuine typos fail, but the $1mm
   per-segment rounding in the 10-Q allocation table does not).
2. Equity ties: net-equity total == total stockholders' equity.
3. Book value reconciliation is internally consistent
   (economic = GAAP common + FV add-backs).
4. Baseline BVPS (zero shock) reproduces the reported GAAP $12.70 and
   Economic $13.22 to the penny — the single most important check.

Run standalone:  python validate.py --config config/2026Q1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


class ConfigValidationError(Exception):
    """Raised when the config fails a hard cross-footing / tie-out check."""


# --- tolerances --------------------------------------------------------------
# Segment figures in the allocation table are rounded to the nearest $1mm, so a
# column of N segments can drift by up to ~0.5*N mm from pure rounding. We also
# allow 0.5% of the column total for large columns. A real transposition/typo
# in a small column exceeds this and fails.
def _xfoot_tol_mm(total_mm: float, n_segments: int) -> float:
    return max(0.5 * n_segments, 0.005 * abs(total_mm))


PENNY = 0.005          # BVPS must round to reported value at 2 dp
RECON_TOL_MM = 0.1     # for figures the 10-Q reports to one decimal ($mm)
EQUITY_TOL_MM = 0.5    # net-equity total vs stockholders' equity ($mm)


class _Report:
    """Accumulates pass/fail lines; raises with a full report on any failure."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failed = True
        self.lines.append(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))

    def note(self, text: str) -> None:
        self.lines.append(f"  [NOTE] {text}")

    def finish(self, header: str) -> None:
        body = "\n".join([header, *self.lines])
        if self.failed:
            raise ConfigValidationError(
                body + "\n\nConfig REJECTED — fix the failing line(s) above."
            )
        # success: print the clean report so the user sees the tie-outs
        print(body)
        print("  ---> config validated OK\n")


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_and_validate(path: str | Path, verbose: bool = True) -> dict:
    """Load YAML, run all cross-footing checks, return the config dict.

    Raises ConfigValidationError (with a full report) on any hard failure.
    """
    cfg = _load_yaml(path)
    rep = _Report()

    bs = cfg["balance_sheet"]
    aa = cfg["asset_allocation"]
    segs = aa["segments"]
    totals = aa["totals_mm"]
    bvr = cfg["book_value_reconciliation"]
    n = len(segs)

    # --- 1. financing columns cross-foot to reported totals ------------------
    columns = {
        "assets_mm": "assets",
        "non_mtm_debt_mm": "non_mtm_debt",
        "mtm_debt_mm": "mtm_debt",
        "securitized_debt_mm": "securitized_debt",
        "senior_other_mm": "senior_other",
        "net_equity_mm": "net_equity",
    }
    for seg_field, tot_key in columns.items():
        col_sum = sum(s[seg_field] for s in segs)
        reported = totals[tot_key]
        tol = _xfoot_tol_mm(reported, n)
        resid = col_sum - reported
        rep.check(
            abs(resid) <= tol,
            f"column '{tot_key}' sums to reported total",
            f"sum={col_sum:.1f} reported={reported:.1f} "
            f"residual={resid:+.1f}mm tol=+/-{tol:.1f}mm",
        )

    # --- 2. equity ties ------------------------------------------------------
    equity_mm = bs["total_stockholders_equity"] / 1000.0
    net_equity_total_mm = totals["net_equity"]
    rep.check(
        abs(net_equity_total_mm - equity_mm) <= max(EQUITY_TOL_MM, 0.005 * equity_mm),
        "net-equity total ties to total stockholders' equity",
        f"net_equity={net_equity_total_mm:.1f} equity={equity_mm:.1f}mm",
    )

    # preferred split ties to total preference
    pref_sum = bs["preferred_series_b"] + bs["preferred_series_c"]
    rep.check(
        pref_sum == bs["preferred_liquidation_preference"],
        "Series B + C == preferred liquidation preference",
        f"B+C={pref_sum} pref={bs['preferred_liquidation_preference']}",
    )

    # --- 3. book value reconciliation is internally consistent ---------------
    gaap_common_mm = (
        bs["total_stockholders_equity"] - bs["preferred_liquidation_preference"]
    ) / 1000.0
    rep.check(
        abs(gaap_common_mm - bvr["gaap_common_equity_for_bvps_mm"]) <= RECON_TOL_MM,
        "GAAP common equity == equity - preferred",
        f"computed={gaap_common_mm:.1f} reported={bvr['gaap_common_equity_for_bvps_mm']:.1f}mm",
    )
    econ_recon_mm = (
        bvr["gaap_common_equity_for_bvps_mm"]
        + bvr["fv_adj_residential_loans_carrying_mm"]
        + bvr["fv_adj_securitized_debt_carrying_mm"]
    )
    rep.check(
        abs(econ_recon_mm - bvr["economic_common_equity_mm"]) <= RECON_TOL_MM,
        "Economic common equity == GAAP common + FV add-backs",
        f"computed={econ_recon_mm:.1f} reported={bvr['economic_common_equity_mm']:.1f}mm",
    )

    # --- 4. baseline BVPS reproduces reported figures TO THE PENNY -----------
    shares = bs["common_shares_outstanding"]  # thousands
    # compute directly from balance-sheet primitives (thousands / thousands = $/sh)
    gaap_bvps = (
        bs["total_stockholders_equity"] - bs["preferred_liquidation_preference"]
    ) / shares
    econ_common_thousands = (
        bs["total_stockholders_equity"]
        - bs["preferred_liquidation_preference"]
        + bvr["fv_adj_residential_loans_carrying_mm"] * 1000.0
        + bvr["fv_adj_securitized_debt_carrying_mm"] * 1000.0
    )
    econ_bvps = econ_common_thousands / shares

    rep.check(
        abs(round(gaap_bvps, 2) - bvr["reported_gaap_bvps"]) < PENNY,
        "baseline GAAP BVPS reproduces reported figure",
        f"computed={gaap_bvps:.4f} -> {round(gaap_bvps,2):.2f} "
        f"reported={bvr['reported_gaap_bvps']:.2f}",
    )
    rep.check(
        abs(round(econ_bvps, 2) - bvr["reported_economic_bvps"]) < PENNY,
        "baseline Economic BVPS reproduces reported figure",
        f"computed={econ_bvps:.4f} -> {round(econ_bvps,2):.2f} "
        f"reported={bvr['reported_economic_bvps']:.2f}",
    )

    # --- loan characteristics present for every credit-sensitive segment -----
    lc = cfg["loan_characteristics"]
    for s in segs:
        if s.get("credit_sensitive"):
            has = s["key"] in lc
            rep.check(has, f"loan characteristics present for '{s['key']}'")

    # --- delinquency aging cross-foots + spread classes present (if present) -
    if "delinquency_aging" in cfg:
        da = cfg["delinquency_aging"]["segments"]
        for key, ag in da.items():
            bucket_sum = (ag["current"] + ag["dpd_30_59"]
                          + ag["dpd_60_89"] + ag["dpd_90plus"])
            rep.check(
                abs(bucket_sum - ag["upb"]) <= 1.0,  # $ thousands, to the dollar
                f"aging buckets cross-foot to UPB for '{key}'",
                f"sum={bucket_sum} upb={ag['upb']} resid={bucket_sum - ag['upb']:+d}",
            )
        if "net_interest_spread" in cfg:
            nis = cfg["net_interest_spread"]
            for key, cls_name in nis["segment_class"].items():
                rep.check(
                    cls_name in nis["classes"] and key in da,
                    f"spread class + aging present for '{key}'",
                    f"class='{cls_name}'",
                )

    # --- 5. interest-rate shock table ties to the filing (if present) --------
    if "interest_rate_shock" in cfg:
        irs = cfg["interest_rate_shock"]
        tbl = irs["reported_shock_table"]
        x = np.array([r["bps"] for r in tbl], dtype=float)
        y = np.array([r["npv_change_thousands"] for r in tbl], dtype=float)
        A = np.vstack([x, x ** 2]).T          # no intercept -> 0bps gives 0
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        a, b = float(coef[0]), float(coef[1])
        fitted = a * x + b * x ** 2
        max_err = float(np.max(np.abs(fitted - y)))
        rep.check(
            max_err <= 1.0,  # ties to within $1k across all anchors
            "rate-shock quadratic (duration+convexity) reproduces reported table",
            f"max |model-reported| = ${max_err:.2f}k across {len(tbl)} anchors",
        )
        # pct_equity column reproduces from dNPV / total equity
        equity_k = bs["total_stockholders_equity"]
        for r in tbl:
            model_pct = r["npv_change_thousands"] / equity_k
            rep.check(
                abs(model_pct - r["pct_equity"]) <= 0.001,
                f"rate shock {int(r['bps']):+d}bps: % equity ties",
                f"model={model_pct*100:+.2f}% reported={r['pct_equity']*100:+.2f}%",
            )
        # component market values cross-foot to total assets
        comp = irs["components"]
        mv_sum = (comp["residential_whole_loans"]["market_value_mm"]
                  + comp["securities"]["market_value_mm"]
                  + comp["other_and_cash"]["market_value_mm"])
        rep.check(
            abs(mv_sum - totals["assets"]) <= _xfoot_tol_mm(totals["assets"], n),
            "duration bucket market values sum to total assets",
            f"sum={mv_sum:.0f} assets={totals['assets']:.0f}mm",
        )

    header = f"Validating config: {Path(path).name}  (as of {cfg['meta']['as_of_date']})"
    if verbose:
        rep.finish(header)
    else:
        # still raise on failure, but stay quiet on success
        if rep.failed:
            rep.finish(header)

    # stash a couple of derived, validated values for downstream convenience
    cfg["_derived"] = {
        "gaap_bvps_baseline": round(gaap_bvps, 4),
        "econ_bvps_baseline": round(econ_bvps, 4),
        "gaap_common_equity_thousands": bs["total_stockholders_equity"]
        - bs["preferred_liquidation_preference"],
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate an MFA stress config.")
    ap.add_argument("--config", default="config/2026Q1.yaml")
    args = ap.parse_args()
    try:
        load_and_validate(args.config, verbose=True)
    except ConfigValidationError as exc:
        print("\nCONFIG VALIDATION FAILED\n========================", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
