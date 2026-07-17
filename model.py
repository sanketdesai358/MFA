"""
model.py : vectorized NumPy Monte Carlo engine for MFA book-value stress.

Design goals
------------
* 10k+ paths in <2s. No per-path Python loops : the only Python loop is over the
  handful of balance-sheet SEGMENTS (5 credit-sensitive + 2 loss-exempt). Within
  a segment everything is vectorized over (n_sims x n_loans) with float32.
* Deterministic given a seed: the loan tape (LTVs, delinquency flags) is FROZEN
  at the seed; only the home-price draws vary across paths.
* All money is carried in $ MILLIONS internally (to match the allocation table);
  shares are carried in MILLIONS, so BVPS = equity_mm / shares_mm is $/share.

Key modeling judgments (documented; change with intent)
-------------------------------------------------------
* "Cross-segment systematic sigma" is implemented as a COMMON housing factor
  shared by all segments on a given path: segment_shock = user_shock + eps_sys,
  eps_sys ~ N(0, systematic_sigma). At sigma=0 every path sees exactly the user
  shock and only loan-level dispersion varies. This is the sensible reading of
  "systematic" (the whole market moves together) and is what the HP-shock x
  dispersion heatmap sweeps.
* Segment asset value is used as the aggregate UPB proxy for the credit-
  sensitive segments (loans carried near par); each representative loan carries
  EQUAL UPB, so the (UPB-weighted) mean LTV equals the segment WA LTV exactly.
* Non-recourse cap uses retained_equity_in_deals = net_equity * securitized
  financing share as MFA's loss-absorption ceiling on securitized collateral;
  losses beyond that are passed to securitization bondholders.
* Margin call on MTM repo = mtm_debt * economic_loss_rate * mark_severity
  (a collateral mark-down at the original advance rate demands that much cash).
  Agency margin call = |agency_price_shock| * agency_mtm_debt (no credit; the
  $3.1B agency repo typically DOMINATES the liquidity draw, not credit).
* Transitional-segment LTVs in the 10-Q are often as-repaired/stabilized; as-is
  exposure is worse, so the transitional beta support runs to 110% LTV.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

F = np.float32  # big loan-level arrays use float32 for speed + memory


# =============================================================================
# Parameters
# =============================================================================
@dataclass
class ModelParams:
    hp_shock: float
    loan_dispersion_sigma: float
    systematic_sigma: float
    foreclosure_cost: float
    n_sims: int
    agency_price_shock: float          # negative number, e.g. -0.03
    lender_mark_severity: float
    fire_sale_discount: float
    liquidity_haircut: float
    ltv_std_dev: float
    loans_per_segment: int
    seed: int
    default_mode: str                  # "negative_equity" | "ramp"
    ramp_ltv_low: float
    ramp_ltv_high: float
    remark_fv_to_zero: bool
    advance_rate: float = 0.90         # repo advance rate for forced deleveraging
    hpa_cap_up: float = 0.20
    hpa_cap_down: float = -0.90

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "ModelParams":
        d = cfg["simulation_defaults"]
        base = dict(
            hp_shock=d["hp_shock"],
            loan_dispersion_sigma=d["loan_dispersion_sigma"],
            systematic_sigma=d["systematic_sigma"],
            foreclosure_cost=d["foreclosure_cost"],
            n_sims=int(d["n_sims"]),
            agency_price_shock=d["agency_price_shock"],
            lender_mark_severity=d["lender_mark_severity"],
            fire_sale_discount=d["fire_sale_discount"],
            liquidity_haircut=d["liquidity_haircut"],
            ltv_std_dev=d["ltv_std_dev"],
            loans_per_segment=int(d["loans_per_segment"]),
            seed=int(d["seed"]),
            default_mode=d["default_mode"],
            ramp_ltv_low=d["ramp_ltv_low"],
            ramp_ltv_high=d["ramp_ltv_high"],
            remark_fv_to_zero=bool(d["remark_fv_to_zero"]),
            advance_rate=d.get("advance_rate", 0.90),
        )
        base.update(overrides)
        return cls(**base)

    def replace(self, **overrides) -> "ModelParams":
        return replace(self, **overrides)


# =============================================================================
# Results container
# =============================================================================
@dataclass
class SimResults:
    params: ModelParams
    # per-path (n_sims,) arrays -----------------------------------------------
    gaap_bvps: np.ndarray
    econ_bvps: np.ndarray
    loss_to_mfa: np.ndarray          # $mm total (credit post-cap + fire-sale)
    credit_loss_to_mfa: np.ndarray   # $mm credit only, post non-recourse cap
    fire_sale_loss: np.ndarray       # $mm
    losses_to_bondholders: np.ndarray  # $mm passed to securitization investors
    margin_calls: np.ndarray         # $mm total
    stressed_equity: np.ndarray      # $mm
    preferred_impairment: np.ndarray  # $mm
    breach: np.ndarray               # bool
    # per-segment (n_sims, n_seg) matrices for median-path bridge -------------
    seg_keys: list
    seg_names: list
    seg_loss_to_mfa: np.ndarray      # (n_sims, n_seg) $mm
    seg_gross_loss: np.ndarray       # (n_sims, n_seg) $mm
    seg_bondholder: np.ndarray       # (n_sims, n_seg) $mm
    seg_margin: np.ndarray           # (n_sims, n_seg) $mm
    seg_default_rate: np.ndarray     # (n_sims, n_seg)
    seg_net_equity: np.ndarray       # (n_seg,) allocated equity $mm
    # frozen tape samples for LTV density plots -------------------------------
    ltv_tape: dict                   # key -> (pre_ltv array, post_ltv median-path array)
    # scalars -----------------------------------------------------------------
    available_liquidity: float       # $mm
    baseline_gaap_bvps: float
    baseline_econ_bvps: float
    shares_mm: float

    def median_path_index(self) -> int:
        """Index of the path whose Economic BVPS is the median (for the bridge)."""
        return int(np.argsort(self.econ_bvps)[len(self.econ_bvps) // 2])


# =============================================================================
# Loan-tape construction (frozen at seed)
# =============================================================================
def _beta_ltv_tape(
    rng: np.random.Generator,
    n: int,
    mean: float,
    std: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Draw n LTVs from a Beta scaled to [lo, hi] with the given mean and std.

    Solves the Beta(a,b) moments so the support-normalized mean/variance match;
    clamps std if it exceeds the max achievable for the target mean.
    """
    rng_span = hi - lo
    mu = (mean - lo) / rng_span                    # normalized target mean in (0,1)
    mu = min(max(mu, 1e-3), 1 - 1e-3)
    sigma = std / rng_span
    # max variance on [0,1] for this mean is mu*(1-mu); keep strictly below it
    max_sigma2 = mu * (1 - mu) * 0.98
    sigma2 = min(sigma * sigma, max_sigma2)
    nu = mu * (1 - mu) / sigma2 - 1.0              # a + b
    nu = max(nu, 1e-3)
    a = mu * nu
    b = (1 - mu) * nu
    x = rng.beta(a, b, size=n)
    return (lo + rng_span * x).astype(F)


def _build_tape(cfg: dict, params: ModelParams) -> dict:
    """Build and freeze the per-segment loan tape (LTVs + delinquency flags)."""
    rng = np.random.default_rng(params.seed)
    lc = cfg["loan_characteristics"]
    tape = {}
    for seg in cfg["asset_allocation"]["segments"]:
        if not seg.get("credit_sensitive"):
            continue
        key = seg["key"]
        ch = lc[key]
        n = params.loans_per_segment
        transitional = seg.get("transitional", False)
        lo, hi = (0.10, 1.10) if transitional else (0.10, 1.00)
        ltv = _beta_ltv_tape(rng, n, ch["wa_ltv"], params.ltv_std_dev, lo, hi)
        # freeze delinquency: a fixed fraction (60+ DPD) always defaults
        n_delinq = int(round(ch["dpd_60plus"] * n))
        delinq = np.zeros(n, dtype=bool)
        if n_delinq > 0:
            idx = rng.choice(n, size=n_delinq, replace=False)
            delinq[idx] = True
        upb_total = float(seg["assets_mm"])         # UPB proxy = segment assets
        upb_per_loan = upb_total / n
        home_value = (upb_per_loan / ltv).astype(F)  # original home value per loan
        tape[key] = dict(
            ltv=ltv,
            delinq=delinq,
            upb_per_loan=upb_per_loan,
            upb_total=upb_total,
            home_value=home_value,
            dpd=ch["dpd_60plus"],
        )
    return tape


# =============================================================================
# Main simulation
# =============================================================================
def run_simulation(cfg: dict, params: ModelParams) -> SimResults:
    rng = np.random.default_rng(params.seed + 1)  # +1: independent of tape RNG
    n_sims = params.n_sims

    bs = cfg["balance_sheet"]
    bvr = cfg["book_value_reconciliation"]
    equity_mm = bs["total_stockholders_equity"] / 1000.0
    pref_mm = bs["preferred_liquidation_preference"] / 1000.0
    shares_mm = bs["common_shares_outstanding"] / 1000.0
    fv_add = 0.0 if params.remark_fv_to_zero else (
        bvr["fv_adj_residential_loans_carrying_mm"]
        + bvr["fv_adj_securitized_debt_carrying_mm"]
    )

    tape = _build_tape(cfg, params)

    # common systematic housing factor per path (shared across segments)
    eps_sys = rng.normal(0.0, params.systematic_sigma, size=n_sims).astype(F)

    segs = cfg["asset_allocation"]["segments"]
    credit_segs = [s for s in segs if s.get("credit_sensitive")]
    n_seg = len(segs)

    # accumulators
    seg_keys, seg_names = [], []
    seg_loss_to_mfa = np.zeros((n_sims, n_seg), dtype=F)
    seg_gross_loss = np.zeros((n_sims, n_seg), dtype=F)
    seg_bondholder = np.zeros((n_sims, n_seg), dtype=F)
    seg_margin = np.zeros((n_sims, n_seg), dtype=F)
    seg_default_rate = np.zeros((n_sims, n_seg), dtype=F)
    seg_net_equity = np.zeros(n_seg, dtype=F)

    total_credit_loss = np.zeros(n_sims, dtype=F)
    total_margin = np.zeros(n_sims, dtype=F)
    total_bondholder = np.zeros(n_sims, dtype=F)
    ltv_tape_out: dict = {}

    # we need the median path to sample post-shock LTVs; do a first pass to get
    # econ BVPS, then re-derive median-path LTVs from stored per-segment HPA is
    # expensive : instead store post-shock LTV for the path nearest to the median
    # SEGMENT shock, which is representative. Simpler: store post-shock LTV at the
    # path whose systematic factor is the median (deterministic, representative).
    median_sys_path = int(np.argsort(eps_sys)[n_sims // 2])

    for j, seg in enumerate(segs):
        key = seg["key"]
        seg_keys.append(key)
        seg_names.append(seg["name"])
        seg_net_equity[j] = seg["net_equity_mm"]

        # ---- loss-exempt segments (Agency, Other): margin only --------------
        if not seg.get("credit_sensitive"):
            if seg.get("agency"):
                # deterministic across paths: price-shock mark on agency MTM repo
                agency_margin = abs(params.agency_price_shock) * seg["mtm_debt_mm"]
                seg_margin[:, j] = agency_margin
                total_margin += agency_margin
            # Other, net: credit-loss-exempt, no specified price shock -> no margin
            continue

        # ---- credit-sensitive segment: full loan-level simulation -----------
        t = tape[key]
        n_loans = params.loans_per_segment
        upb_pl = t["upb_per_loan"]
        upb_total = t["upb_total"]
        home_value = t["home_value"]              # (n_loans,)
        ltv_pre = t["ltv"]                        # (n_loans,)

        seg_shock = (params.hp_shock + eps_sys).astype(F)          # (n_sims,)
        # loan HPA ~ N(segment_shock, dispersion), capped
        hpa = seg_shock[:, None] + params.loan_dispersion_sigma * rng.standard_normal(
            (n_sims, n_loans)
        ).astype(F)
        np.clip(hpa, params.hpa_cap_down, params.hpa_cap_up, out=hpa)

        post_value = home_value[None, :] * (1.0 + hpa)             # (n_sims,n_loans)
        post_ltv = upb_pl / np.maximum(post_value, F(1e-6))

        # default indicator
        if params.default_mode == "ramp":
            p = (post_ltv - F(params.ramp_ltv_low)) / F(
                params.ramp_ltv_high - params.ramp_ltv_low
            )
            np.clip(p, 0.0, 1.0, out=p)
            u = rng.random((n_sims, n_loans), dtype=F)
            default_ind = u < p
        else:  # negative_equity
            default_ind = post_ltv > 1.0
        # 60+ DPD floor: delinquent loans default regardless of equity
        default_ind |= t["delinq"][None, :]

        # loss given default, capped at UPB, credit-exposed loans only
        fc = F(params.foreclosure_cost)
        recovery = post_value * (1.0 - fc)
        loss_pl = np.where(default_ind, upb_pl - recovery, F(0.0))
        np.clip(loss_pl, 0.0, upb_pl, out=loss_pl)

        gross = loss_pl.sum(axis=1)                                # (n_sims,) $mm
        np.minimum(gross, F(upb_total), out=gross)                 # <= segment assets
        drate = default_ind.mean(axis=1).astype(F)

        seg_gross_loss[:, j] = gross
        seg_default_rate[:, j] = drate

        # ---- non-recourse securitization cap --------------------------------
        fin_total = (
            seg["non_mtm_debt_mm"] + seg["mtm_debt_mm"]
            + seg["securitized_debt_mm"] + seg["senior_other_mm"]
        )
        sec_share = seg["securitized_debt_mm"] / fin_total if fin_total > 0 else 0.0
        retained_equity = seg["net_equity_mm"] * sec_share         # MFA's cap
        loss_sec = gross * F(sec_share)
        loss_nonsec = gross - loss_sec
        mfa_sec = np.minimum(loss_sec, F(retained_equity))
        to_bond = loss_sec - mfa_sec
        loss_to_mfa_seg = loss_nonsec + mfa_sec

        seg_bondholder[:, j] = to_bond
        seg_loss_to_mfa[:, j] = loss_to_mfa_seg
        total_credit_loss += loss_to_mfa_seg
        total_bondholder += to_bond

        # ---- margin call on MTM collateral ----------------------------------
        econ_loss_rate = gross / F(upb_total)
        margin_seg = F(seg["mtm_debt_mm"]) * econ_loss_rate * F(params.lender_mark_severity)
        seg_margin[:, j] = margin_seg
        total_margin += margin_seg

        # ---- store representative (median-systematic-path) LTVs for plotting -
        ltv_tape_out[key] = (
            np.asarray(ltv_pre, dtype=float),
            np.asarray(post_ltv[median_sys_path], dtype=float),
        )

    # =========================================================================
    # Liquidity test & forced deleveraging
    # =========================================================================
    available_liq = (
        bs["cash_unrestricted"] / 1000.0
        + bvr["other_unencumbered_securities_mm"] * (1.0 - params.liquidity_haircut)
    )
    shortfall = np.clip(total_margin - F(available_liq), 0.0, None)
    breach = shortfall > 0
    forced_delever = shortfall / F(max(1e-6, 1.0 - params.advance_rate))
    fire_sale_loss = forced_delever * F(params.fire_sale_discount)

    # =========================================================================
    # Book value waterfall
    # =========================================================================
    loss_to_mfa = total_credit_loss + fire_sale_loss
    stressed_equity = F(equity_mm) - loss_to_mfa

    # common equity, with preferred impairment when equity < preference
    excess_over_pref = stressed_equity - F(pref_mm)
    common_equity = np.clip(excess_over_pref, 0.0, None)
    preferred_impairment = np.clip(F(pref_mm) - stressed_equity, 0.0, None)

    gaap_bvps = common_equity / F(shares_mm)
    econ_common = stressed_equity + F(fv_add) - F(pref_mm)
    econ_bvps = np.clip(econ_common, 0.0, None) / F(shares_mm)

    baseline_gaap = (equity_mm - pref_mm) / shares_mm
    baseline_econ = (
        equity_mm - pref_mm
        + bvr["fv_adj_residential_loans_carrying_mm"]
        + bvr["fv_adj_securitized_debt_carrying_mm"]
    ) / shares_mm

    return SimResults(
        params=params,
        gaap_bvps=np.asarray(gaap_bvps, dtype=float),
        econ_bvps=np.asarray(econ_bvps, dtype=float),
        loss_to_mfa=np.asarray(loss_to_mfa, dtype=float),
        credit_loss_to_mfa=np.asarray(total_credit_loss, dtype=float),
        fire_sale_loss=np.asarray(fire_sale_loss, dtype=float),
        losses_to_bondholders=np.asarray(total_bondholder, dtype=float),
        margin_calls=np.asarray(total_margin, dtype=float),
        stressed_equity=np.asarray(stressed_equity, dtype=float),
        preferred_impairment=np.asarray(preferred_impairment, dtype=float),
        breach=np.asarray(breach, dtype=bool),
        seg_keys=seg_keys,
        seg_names=seg_names,
        seg_loss_to_mfa=np.asarray(seg_loss_to_mfa, dtype=float),
        seg_gross_loss=np.asarray(seg_gross_loss, dtype=float),
        seg_bondholder=np.asarray(seg_bondholder, dtype=float),
        seg_margin=np.asarray(seg_margin, dtype=float),
        seg_default_rate=np.asarray(seg_default_rate, dtype=float),
        seg_net_equity=np.asarray(seg_net_equity, dtype=float),
        ltv_tape=ltv_tape_out,
        available_liquidity=float(available_liq),
        baseline_gaap_bvps=round(baseline_gaap, 4),
        baseline_econ_bvps=round(baseline_econ, 4),
        shares_mm=shares_mm,
    )


# =============================================================================
# Summary helpers (used by the dashboard KPI row & narrative)
# =============================================================================
def summarize(res: SimResults, cfg: dict) -> dict:
    equity_mm = cfg["balance_sheet"]["total_stockholders_equity"] / 1000.0
    exp_loss = float(res.loss_to_mfa.mean())
    return dict(
        median_econ_bvps=float(np.median(res.econ_bvps)),
        median_gaap_bvps=float(np.median(res.gaap_bvps)),
        p5_econ_bvps=float(np.percentile(res.econ_bvps, 5)),
        p95_econ_bvps=float(np.percentile(res.econ_bvps, 95)),
        p5_gaap_bvps=float(np.percentile(res.gaap_bvps, 5)),
        p95_gaap_bvps=float(np.percentile(res.gaap_bvps, 95)),
        expected_loss_mm=exp_loss,
        expected_loss_pct_equity=exp_loss / equity_mm,
        p_breach=float(res.breach.mean()),
        # common wipeout gauged on Economic common equity (the headline measure)
        p_common_wiped=float((res.econ_bvps <= 1e-9).mean()),
        p_pref_impaired=float((res.preferred_impairment > 1e-9).mean()),
        baseline_econ_bvps=res.baseline_econ_bvps,
        baseline_gaap_bvps=res.baseline_gaap_bvps,
        available_liquidity=res.available_liquidity,
        median_margin_call=float(np.median(res.margin_calls)),
    )


def segment_table(res: SimResults, cfg: dict) -> list:
    """Per-segment mean diagnostics for the dashboard table."""
    rows = []
    for j, key in enumerate(res.seg_keys):
        gross = res.seg_gross_loss[:, j]
        drate = res.seg_default_rate[:, j]
        # severity = gross loss / defaulted UPB (guard divide-by-zero)
        seg_cfg = next(s for s in cfg["asset_allocation"]["segments"] if s["key"] == key)
        upb = seg_cfg["assets_mm"]
        defaulted_upb = np.maximum(drate * upb, 1e-9)
        severity = np.where(drate > 1e-6, gross / defaulted_upb, 0.0)
        rows.append(dict(
            segment=res.seg_names[j],
            key=key,
            credit_sensitive=bool(seg_cfg.get("credit_sensitive")),
            net_equity_mm=float(res.seg_net_equity[j]),
            default_rate=float(np.mean(drate)),
            severity=float(np.mean(severity)),
            gross_loss_mm=float(np.mean(gross)),
            loss_to_mfa_mm=float(np.mean(res.seg_loss_to_mfa[:, j])),
            loss_to_bondholders_mm=float(np.mean(res.seg_bondholder[:, j])),
            margin_call_mm=float(np.mean(res.seg_margin[:, j])),
            equity_remaining_mm=float(res.seg_net_equity[j] - np.mean(res.seg_loss_to_mfa[:, j])),
        ))
    return rows


def run_heatmap(
    cfg: dict,
    base_params: ModelParams,
    shocks: np.ndarray,
    dispersions: np.ndarray,
    n_sims_cell: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid of (hp_shock x dispersion) -> (median econ BVPS, P(breach)).

    Returns two 2-D arrays shaped (len(dispersions), len(shocks)) so rows are
    dispersion (y-axis) and columns are shock (x-axis) for a Plotly heatmap.
    """
    med = np.zeros((len(dispersions), len(shocks)))
    pbr = np.zeros((len(dispersions), len(shocks)))
    for r, disp in enumerate(dispersions):
        for c, shock in enumerate(shocks):
            p = base_params.replace(
                hp_shock=float(shock),
                loan_dispersion_sigma=float(disp),
                n_sims=int(n_sims_cell),
            )
            res = run_simulation(cfg, p)
            med[r, c] = float(np.median(res.econ_bvps))
            pbr[r, c] = float(res.breach.mean())
    return med, pbr


# =============================================================================
# Interest-rate / duration shock (10-Q "Shock Table")
# =============================================================================
def _fit_npv_quadratic(cfg: dict) -> tuple[float, float]:
    """Fit dNPV($ thousands) = a*bps + b*bps^2 to the reported shock table.

    MFA's disclosed figures are exactly a duration+convexity quadratic in the
    rate shock, so this reproduces the filing to the dollar (0 intercept: a 0bp
    shock is a 0 change). Returns (a, b) in $ thousands per bp / per bp^2.
    """
    tbl = cfg["interest_rate_shock"]["reported_shock_table"]
    x = np.array([r["bps"] for r in tbl], dtype=float)
    y = np.array([r["npv_change_thousands"] for r in tbl], dtype=float)
    A = np.vstack([x, x ** 2]).T          # no constant term -> passes through 0
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def rate_shock_curve(cfg: dict, bps) -> dict:
    """Interest-rate shock analysis over an array of parallel bps shocks.

    Returns, per bps point:
      * dnpv_mm       : net portfolio value change (incl. swaps & debt), $mm,
                        from the quadratic fit to the reported table (ties to 10-Q)
      * loan_pnl_mm   : mark-to-market P&L on the residential whole-loan book
                        alone, from disclosed loan duration/convexity, $mm
      * hedge_other_mm : implied non-loan/hedge contribution = dnpv - loan_pnl
      * econ_bvps_net       : Economic BVPS applying the net (hedged) change
      * econ_bvps_loans_only : Economic BVPS if ONLY the loan book re-marked
      * pct_equity    : dnpv / total stockholders' equity
    """
    irs = cfg["interest_rate_shock"]
    bs = cfg["balance_sheet"]
    bvr = cfg["book_value_reconciliation"]
    equity_mm = bs["total_stockholders_equity"] / 1000.0
    shares_mm = bs["common_shares_outstanding"] / 1000.0
    econ_common_mm = bvr["economic_common_equity_mm"]

    a, b = _fit_npv_quadratic(cfg)                     # $000 per bp / bp^2
    bps = np.atleast_1d(np.asarray(bps, dtype=float))

    dnpv_mm = (a * bps + b * bps ** 2) / 1000.0        # -> $mm, net of hedges

    ln = irs["components"]["residential_whole_loans"]
    dy = bps / 10000.0                                 # bps -> decimal yield
    loan_pct = -ln["duration"] * dy + 0.5 * ln["convexity"] * dy ** 2
    loan_pnl_mm = loan_pct * ln["market_value_mm"]

    return dict(
        bps=bps,
        dnpv_mm=dnpv_mm,
        loan_pnl_mm=loan_pnl_mm,
        hedge_other_mm=dnpv_mm - loan_pnl_mm,
        econ_bvps_net=(econ_common_mm + dnpv_mm) / shares_mm,
        econ_bvps_loans_only=(econ_common_mm + loan_pnl_mm) / shares_mm,
        pct_equity=dnpv_mm / equity_mm,
        a=a, b=b,
        baseline_econ_bvps=econ_common_mm / shares_mm,
        econ_common_mm=econ_common_mm,
        shares_mm=shares_mm,
    )


def rate_shock_table(cfg: dict) -> list:
    """Model-reproduced Shock Table at the reported anchor points, for tie-out."""
    tbl = cfg["interest_rate_shock"]["reported_shock_table"]
    npv_base_mm = cfg["interest_rate_shock"]["net_portfolio_value_base_mm"]
    bps = np.array([r["bps"] for r in tbl], dtype=float)
    cur = rate_shock_curve(cfg, bps)
    rows = []
    for i, r in enumerate(tbl):
        model_npv_000 = cur["dnpv_mm"][i] * 1000.0
        rows.append(dict(
            bps=int(r["bps"]),
            model_npv_change_thousands=model_npv_000,
            reported_npv_change_thousands=r["npv_change_thousands"],
            model_pct_npv=cur["dnpv_mm"][i] / npv_base_mm,
            reported_pct_npv=r["pct_npv"],
            model_pct_equity=float(cur["pct_equity"][i]),
            reported_pct_equity=r["pct_equity"],
            econ_bvps=float(cur["econ_bvps_net"][i]),
        ))
    return rows


# ---------------------------------------------------------------------------
# Debt-service coverage (DSCR) under delinquency stress (10-Q p.66 + p.18)
# ---------------------------------------------------------------------------
def _delinq_rate(ag: dict, measure: str) -> float:
    """Delinquency rate for a segment's aging row under a chosen DPD measure."""
    upb = ag["upb"]
    if upb <= 0:
        return 0.0
    if measure == "dpd_30plus":
        num = ag["dpd_30_59"] + ag["dpd_60_89"] + ag["dpd_90plus"]
    elif measure == "dpd_90plus":
        num = ag["dpd_90plus"]
    else:  # dpd_60plus (default)
        num = ag["dpd_60_89"] + ag["dpd_90plus"]
    return num / upb


def dscr_analysis(cfg: dict, measure: str = "dpd_60plus",
                  include_swap: bool = True, delinq_mult: float = 1.0,
                  recovery_on_delinquent: float = 0.0) -> dict:
    """Debt-service coverage of the residential whole-loan book under a chosen
    delinquency measure.

    For each credit segment:
        interest income  = net_yield        * assets            ($mm / yr)
        interest expense = cost_of_funding  * debt              ($mm / yr)
        net swap carry   = swap_carry       * assets  (optional)

    Delinquent loans are assumed to stop paying interest (less any partial
    `recovery_on_delinquent` on the delinquent slice), so performing income is
    scaled by (1 - delinquency). Debt service is unchanged : MFA still owes its
    lenders regardless of borrower delinquency.

        DSCR = (income * performing_fraction + swap) / expense

    `delinq_mult` scales the observed delinquency rate (stress factor).
    Returns per-segment rows plus portfolio aggregates.
    """
    nis = cfg["net_interest_spread"]
    classes = nis["classes"]
    seg_class = nis["segment_class"]
    aging = cfg["delinquency_aging"]["segments"]
    segs = {s["key"]: s for s in cfg["asset_allocation"]["segments"]}

    rows = []
    tot_income_perf = tot_income_base = tot_swap = tot_expense = 0.0
    for key, cls_name in seg_class.items():
        cls = classes[cls_name]
        seg = segs[key]
        ag = aging[key]
        assets = float(seg["assets_mm"])
        debt = float(seg["non_mtm_debt_mm"] + seg["mtm_debt_mm"]
                     + seg["securitized_debt_mm"] + seg["senior_other_mm"])

        delinq = min(1.0, _delinq_rate(ag, measure) * delinq_mult)
        performing = 1.0 - delinq * (1.0 - recovery_on_delinquent)

        income = cls["net_yield"] * assets
        swap = cls["swap_carry"] * assets if include_swap else 0.0
        expense = cls["cost_of_funding"] * debt
        income_perf = income * performing

        dscr = (income_perf + swap) / expense if expense > 0 else float("inf")
        base_dscr = (income + swap) / expense if expense > 0 else float("inf")

        rows.append(dict(
            key=key, name=seg["name"], assets_mm=assets, debt_mm=debt,
            net_yield=cls["net_yield"], cost_of_funding=cls["cost_of_funding"],
            swap_carry=cls["swap_carry"], delinquency=delinq,
            performing=performing, income_mm=income, income_perf_mm=income_perf,
            swap_mm=swap, expense_mm=expense, dscr=dscr, base_dscr=base_dscr,
        ))
        tot_income_perf += income_perf
        tot_income_base += income
        tot_swap += swap
        tot_expense += expense

    agg = (tot_income_perf + tot_swap) / tot_expense if tot_expense > 0 else float("inf")
    agg_base = (tot_income_base + tot_swap) / tot_expense if tot_expense > 0 else float("inf")
    return dict(
        rows=rows, measure=measure, agg_dscr=agg, agg_base_dscr=agg_base,
        tot_income_perf_mm=tot_income_perf, tot_income_base_mm=tot_income_base,
        tot_swap_mm=tot_swap, tot_expense_mm=tot_expense,
        as_of_aging=cfg["delinquency_aging"].get("as_of"),
        as_of_spread=nis.get("as_of"),
    )


if __name__ == "__main__":
    # quick self-test / benchmark
    import time
    from validate import load_and_validate

    cfg = load_and_validate("config/2026Q1.yaml", verbose=False)
    params = ModelParams.from_config(cfg)
    t0 = time.perf_counter()
    res = run_simulation(cfg, params)
    dt = time.perf_counter() - t0
    s = summarize(res, cfg)
    print(f"ran {params.n_sims:,} paths in {dt*1000:.0f} ms")
    print(f"baseline  GAAP ${res.baseline_gaap_bvps:.2f}  Econ ${res.baseline_econ_bvps:.2f}")
    print(f"median stressed Econ BVPS  ${s['median_econ_bvps']:.2f}")
    print(f"P5/P95 Econ  ${s['p5_econ_bvps']:.2f} / ${s['p95_econ_bvps']:.2f}")
    print(f"expected loss ${s['expected_loss_mm']:.0f}mm ({s['expected_loss_pct_equity']*100:.1f}% of equity)")
    print(f"P(liquidity breach) {s['p_breach']*100:.1f}%   available liq ${s['available_liquidity']:.0f}mm")
    print(f"P(common wiped) {s['p_common_wiped']*100:.2f}%   P(pref impaired) {s['p_pref_impaired']*100:.2f}%")
