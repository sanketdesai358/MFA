"""
app.py: MFA Financial book-value stress dashboard (Streamlit + Plotly).

Run:
    streamlit run app.py -- --config config/2026Q1.yaml

Everything quarter-specific comes from the YAML config; nothing is hard-coded
here. The config is validated on load (validate.py) and the app refuses to run
against a config that does not cross-foot / tie to reported BVPS.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import (
    ModelParams,
    run_simulation,
    run_heatmap,
    segment_table,
    summarize,
    rate_shock_curve,
    rate_shock_table,
    dscr_analysis,
)
from validate import ConfigValidationError, load_and_validate

# ---- institutional light palette -------------------------------------------
BG = "#ffffff"
PANEL = "#f4f6f9"        # light plot area
TEXT = "#1a1f29"
ACCENT = "#2b6cb0"       # blue
GOOD = "#2f9e44"         # green
WARN = "#b7791f"         # amber
BAD = "#e03131"          # red
MUTED = "#5a6474"
SEG_COLORS = ["#2b6cb0", "#2f9e44", "#b7791f", "#c2255c", "#7048e8", "#e8590c", "#495057"]
PLOTLY_TEMPLATE = "plotly_white"


# =============================================================================
# Config / CLI plumbing
# =============================================================================
def _parse_config_path() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/2026Q1.yaml")
    # streamlit passes script args after `--`; ignore unknowns
    args, _ = ap.parse_known_args()
    return args.config


@st.cache_data(show_spinner=False)
def _load_config(path: str, mtime: float) -> dict:
    # mtime in the cache key busts the cache when the YAML changes on disk
    return load_and_validate(path, verbose=False)


@st.cache_data(show_spinner=False)
def _run(path: str, mtime: float, pkey: tuple) -> dict:
    cfg = _load_config(path, mtime)
    params = ModelParams(**dict(pkey))
    res = run_simulation(cfg, params)
    return dict(
        summary=summarize(res, cfg),
        seg_table=segment_table(res, cfg),
        econ_bvps=res.econ_bvps,
        gaap_bvps=res.gaap_bvps,
        margin_calls=res.margin_calls,
        available_liquidity=res.available_liquidity,
        seg_loss_to_mfa=res.seg_loss_to_mfa,
        seg_names=res.seg_names,
        seg_net_equity=res.seg_net_equity,
        fire_sale_loss=res.fire_sale_loss,
        ltv_tape=res.ltv_tape,
        median_path=res.median_path_index(),
        baseline_econ=res.baseline_econ_bvps,
        baseline_gaap=res.baseline_gaap_bvps,
        shares_mm=res.shares_mm,
        fv_add=(0.0 if params.remark_fv_to_zero else (
            cfg["book_value_reconciliation"]["fv_adj_residential_loans_carrying_mm"]
            + cfg["book_value_reconciliation"]["fv_adj_securitized_debt_carrying_mm"])),
    )


@st.cache_data(show_spinner=True)
def _heatmap(path: str, mtime: float, pkey: tuple, shocks: tuple, disps: tuple,
             n_cell: int) -> tuple:
    cfg = _load_config(path, mtime)
    base = ModelParams(**dict(pkey))
    med, pbr = run_heatmap(cfg, base, np.array(shocks), np.array(disps), n_cell)
    return med, pbr


# =============================================================================
# Small chart helpers
# =============================================================================
def _price_to_book(price: float, bvps: float) -> float:
    """Price / Book ratio (e.g. $9.48 / $13.22 = 0.717 -> shown as 72%).

    Below 100% the stock trades under book (a discount); above 100% it trades
    over book (a premium).
    """
    if bvps <= 0:
        return float("nan")
    return price / bvps


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=BG, plot_bgcolor=PANEL,
        height=height, margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        title=dict(text=title or ""),  # "" (not None) avoids an "undefined" label
        font=dict(color=TEXT), legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def bvps_hist(vals: np.ndarray, reported: float, label: str) -> go.Figure:
    med = float(np.median(vals))
    p5 = float(np.percentile(vals, 5))
    fig = go.Figure()
    fig.add_histogram(x=vals, nbinsx=70, marker_color=ACCENT, opacity=0.85,
                      name=label)
    fig.add_vline(x=reported, line=dict(color=GOOD, width=2, dash="dash"),
                  annotation_text=f"reported ${reported:.2f}", annotation_position="top")
    fig.add_vline(x=med, line=dict(color=WARN, width=2),
                  annotation_text=f"median ${med:.2f}", annotation_position="top left")
    fig.add_vline(x=p5, line=dict(color=BAD, width=2, dash="dot"),
                  annotation_text=f"P5 ${p5:.2f}", annotation_position="bottom")
    fig.update_xaxes(title=f"stressed {label} ($/share)")
    fig.update_yaxes(title="paths")
    return _style(fig, 380, f"Stressed {label} distribution")


def bridge_waterfall(d: dict, remark: bool) -> go.Figure:
    path = d["median_path"]
    shares = d["shares_mm"]
    base = d["baseline_econ"]
    seg_loss_ps = d["seg_loss_to_mfa"][path] / shares      # per-share by segment
    fire_ps = d["fire_sale_loss"][path] / shares
    names = d["seg_names"]

    measures, xs, ys, texts = ["absolute"], ["Reported<br>Economic BVPS"], [base], [f"${base:.2f}"]
    for nm, loss in zip(names, seg_loss_ps):
        if loss <= 1e-9:
            continue
        measures.append("relative")
        xs.append(nm)
        ys.append(-loss)
        texts.append(f"-${loss:.2f}")
    if remark:
        # stressed Economic BV zeroes the FV add-backs; show it as an explicit
        # step so the bridge still ties to the (FV-excluded) stressed endpoint.
        fv_ps = d.get("fv_add_baseline", 0.0) / shares
        if fv_ps > 1e-9:
            measures.append("relative")
            xs.append("Re-mark FV<br>to zero")
            ys.append(-fv_ps)
            texts.append(f"-${fv_ps:.2f}")
    if fire_ps > 1e-9:
        measures.append("relative")
        xs.append("Fire-sale<br>(forced delever)")
        ys.append(-fire_ps)
        texts.append(f"-${fire_ps:.2f}")
    measures.append("total")
    xs.append("Stressed<br>Economic BVPS")
    ys.append(None)
    end_val = d["econ_bvps"][path]
    texts.append(f"${end_val:.2f}")

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=xs, y=ys, text=texts,
        textposition="outside", connector=dict(line=dict(color=MUTED)),
        decreasing=dict(marker=dict(color=BAD)),
        increasing=dict(marker=dict(color=GOOD)),
        totals=dict(marker=dict(color=ACCENT)),
    ))
    fig.update_yaxes(title="$/share")
    return _style(fig, 420, "BVPS bridge, median path")


def exp_loss_bars(d: dict) -> go.Figure:
    rows = d["seg_table"]
    names = [r["segment"] for r in rows]
    loss = [r["loss_to_mfa_mm"] for r in rows]
    eq = [r["net_equity_mm"] for r in rows]
    fig = go.Figure()
    fig.add_bar(x=names, y=eq, name="Net equity allocated", marker_color="#30363d")
    fig.add_bar(x=names, y=loss, name="Expected loss to MFA", marker_color=BAD)
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="$mm")
    return _style(fig, 380, "Expected loss to MFA vs net equity")


def margin_liquidity(d: dict) -> go.Figure:
    mc = d["margin_calls"]
    avail = d["available_liquidity"]
    fig = go.Figure()
    fig.add_histogram(x=mc, nbinsx=70, marker_color=ACCENT, opacity=0.85,
                      name="margin calls")
    fig.add_vline(x=avail, line=dict(color=GOOD, width=2),
                  annotation_text=f"available ${avail:.0f}mm", annotation_position="top")
    xmax = float(np.percentile(mc, 99.5))
    if xmax > avail:
        fig.add_vrect(x0=avail, x1=xmax * 1.05, fillcolor=BAD, opacity=0.10,
                      line_width=0, annotation_text="breach", annotation_position="top right")
    fig.update_xaxes(title="total margin calls ($mm)")
    fig.update_yaxes(title="paths")
    return _style(fig, 380, "Margin calls vs available liquidity")


def ltv_densities(d: dict) -> go.Figure:
    fig = go.Figure()
    for i, (key, (pre, post)) in enumerate(d["ltv_tape"].items()):
        c = SEG_COLORS[i % len(SEG_COLORS)]
        fig.add_histogram(x=pre * 100, nbinsx=40, histnorm="probability density",
                          marker_color=c, opacity=0.35, name=f"{key} pre")
        fig.add_histogram(x=post * 100, nbinsx=40, histnorm="probability density",
                          marker_color=c, opacity=0.75, name=f"{key} post")
    fig.add_vline(x=100, line=dict(color=BAD, width=2, dash="dash"),
                  annotation_text="100% LTV")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="LTV (%)", range=[0, 160])
    fig.update_yaxes(title="density")
    return _style(fig, 400, "Pre- vs post-shock LTV by segment")


def heatmap_fig(z: np.ndarray, shocks, disps, title: str, colorscale: str,
                fmt: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{s:.0%}" for s in shocks], y=[f"{dd:.0%}" for dd in disps],
        colorscale=colorscale, colorbar=dict(title=""),
        text=np.vectorize(lambda v: format(v, fmt))(z),
        texttemplate="%{text}", textfont=dict(size=10),
    ))
    fig.update_xaxes(title="HP shock")
    fig.update_yaxes(title="loan dispersion σ")
    return _style(fig, 420, title)


# =============================================================================
# Interest-rate / duration charts (Shock Table)
# =============================================================================
def rate_bvps_curve(cfg: dict, sel_bps: int, use_hedges: bool) -> go.Figure:
    """Economic BVPS vs parallel rate shock, with the reported anchors marked."""
    lo, hi = cfg["interest_rate_shock"]["slider_bps_range"]
    grid = np.linspace(lo, hi, 161)
    cur = rate_shock_curve(cfg, grid)
    y = cur["econ_bvps_net"] if use_hedges else cur["econ_bvps_loans_only"]
    base = cur["baseline_econ_bvps"]

    anchors = cfg["interest_rate_shock"]["reported_shock_table"]
    ax = np.array([r["bps"] for r in anchors], dtype=float)
    acur = rate_shock_curve(cfg, ax)
    ay = acur["econ_bvps_net"] if use_hedges else acur["econ_bvps_loans_only"]

    fig = go.Figure()
    fig.add_scatter(x=grid, y=y, mode="lines", line=dict(color=ACCENT, width=3),
                    name="Economic BVPS")
    fig.add_hline(y=base, line=dict(color=GOOD, width=2, dash="dash"),
                  annotation_text=f"baseline ${base:.2f}", annotation_position="top left")
    fig.add_scatter(x=ax, y=ay, mode="markers",
                    marker=dict(color=WARN, size=10, symbol="diamond"),
                    name="reported anchors (10-Q Shock Table)")
    # selected shock marker
    selcur = rate_shock_curve(cfg, np.array([float(sel_bps)]))
    sy = float((selcur["econ_bvps_net"] if use_hedges
                else selcur["econ_bvps_loans_only"])[0])
    fig.add_vline(x=sel_bps, line=dict(color=BAD, width=2),
                  annotation_text=f"{sel_bps:+d}bps → ${sy:.2f}",
                  annotation_position="bottom right")
    fig.update_xaxes(title="parallel rate shock (bps)")
    fig.update_yaxes(title="Economic BVPS ($/share)")
    return _style(fig, 420,
                  "Economic BVPS vs parallel rate shock"
                  + ("  (net of hedges)" if use_hedges else "  (loans only)"))


def rate_waterfall(cfg: dict, sel_bps: int) -> go.Figure:
    """Bridge from baseline Economic BVPS through loan MTM and hedge/other to net."""
    cur = rate_shock_curve(cfg, np.array([float(sel_bps)]))
    shares = cur["shares_mm"]
    base = cur["baseline_econ_bvps"]
    loan_ps = float(cur["loan_pnl_mm"][0]) / shares
    hedge_ps = float(cur["hedge_other_mm"][0]) / shares
    net_ps = float(cur["econ_bvps_net"][0])

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Baseline<br>Economic BVPS", "Loan MTM<br>P&L",
           "Hedges + other<br>(swaps, debt)", "Net<br>Economic BVPS"],
        y=[base, loan_ps, hedge_ps, None],
        text=[f"${base:.2f}", f"{loan_ps:+.2f}", f"{hedge_ps:+.2f}", f"${net_ps:.2f}"],
        textposition="outside", connector=dict(line=dict(color=MUTED)),
        decreasing=dict(marker=dict(color=BAD)),
        increasing=dict(marker=dict(color=GOOD)),
        totals=dict(marker=dict(color=ACCENT)),
    ))
    fig.update_yaxes(title="$/share")
    return _style(fig, 420, f"BVPS bridge at {sel_bps:+d}bps parallel shock")


# =============================================================================
# DSCR (debt-service coverage) charts + tab body
# =============================================================================
_DSCR_MEASURES = {
    "60+ days past due": "dpd_60plus",
    "30+ days past due": "dpd_30plus",
    "90+ days past due": "dpd_90plus",
}


def dscr_segment_bar(res: dict) -> go.Figure:
    """DSCR by segment at the selected measure, with a 1.0x coverage line."""
    rows = res["rows"]
    names = [r["name"] for r in rows]
    dscr = [r["dscr"] for r in rows]
    base = [r["base_dscr"] for r in rows]
    colors = [BAD if v < 1.0 else (WARN if v < 1.25 else GOOD) for v in dscr]
    fig = go.Figure()
    fig.add_bar(x=names, y=base, name="no-delinquency baseline",
                marker_color="#c7ced8", opacity=0.7)
    fig.add_bar(x=names, y=dscr, name="stressed (selected measure)",
                marker_color=colors,
                text=[f"{v:.2f}x" for v in dscr], textposition="outside")
    fig.add_hline(y=1.0, line=dict(color=BAD, width=2, dash="dash"),
                  annotation_text="1.00x coverage floor",
                  annotation_position="top left")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(title="DSCR (income ÷ debt service)")
    return _style(fig, 420, "DSCR by segment")


def dscr_measures_bar(cfg: dict, include_swap: bool, mult: float) -> go.Figure:
    """Grouped DSCR by segment across all three delinquency measures."""
    fig = go.Figure()
    palette = {"dpd_30plus": WARN, "dpd_60plus": ACCENT, "dpd_90plus": GOOD}
    label = {"dpd_30plus": "30+ DPD", "dpd_60plus": "60+ DPD", "dpd_90plus": "90+ DPD"}
    for m in ("dpd_30plus", "dpd_60plus", "dpd_90plus"):
        res = dscr_analysis(cfg, measure=m, include_swap=include_swap, delinq_mult=mult)
        names = [r["name"] for r in res["rows"]]
        fig.add_bar(x=names, y=[r["dscr"] for r in res["rows"]],
                    name=label[m], marker_color=palette[m])
    fig.add_hline(y=1.0, line=dict(color=BAD, width=2, dash="dash"),
                  annotation_text="1.00x")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="DSCR")
    return _style(fig, 400, "DSCR by segment across delinquency measures")


def _render_dscr_tab(cfg: dict) -> None:
    st.markdown(
        "**Debt-service coverage** of the residential whole-loan book. For each "
        "segment we take interest income (**net yield × assets**, p.66), assume "
        "delinquent loans stop paying, and cover funding cost "
        "(**cost of funding × debt**, p.66):"
    )
    st.latex(r"\text{DSCR}=\frac{\text{net yield}\times\text{assets}\times"
             r"(1-\text{delinquency})\;+\;\text{swap carry}}"
             r"{\text{cost of funding}\times\text{debt}}")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        measure_label = st.radio("Delinquency measure",
                                 list(_DSCR_MEASURES.keys()), index=0)
        measure = _DSCR_MEASURES[measure_label]
    with c2:
        mult = st.slider("Delinquency stress ×", 1.0, 3.0, 1.0, 0.25,
                         help="Scale the observed delinquency rate up to stress coverage.")
        recov = st.slider("Interest recovered on delinquent loans", 0.0, 1.0, 0.0, 0.05,
                          help="0 = delinquent loans pay nothing; 1 = they keep paying.")
    with c3:
        include_swap = st.toggle("Include net swap carry", value=True,
                                 help="Add allocated net swap interest to income.")

    res = dscr_analysis(cfg, measure=measure, include_swap=include_swap,
                        delinq_mult=mult, recovery_on_delinquent=recov)

    # ---- KPI row ------------------------------------------------------------
    k = st.columns(5)
    agg = res["agg_dscr"]
    k[0].metric("Portfolio DSCR", f"{agg:.2f}x",
                f"{agg - res['agg_base_dscr']:+.2f} vs {res['agg_base_dscr']:.2f}x base",
                delta_color="normal")
    k[1].metric("Interest income", f"${res['tot_income_perf_mm']:.0f}mm",
                f"of ${res['tot_income_base_mm']:.0f}mm at par")
    k[2].metric("Net swap carry", f"${res['tot_swap_mm']:.0f}mm")
    k[3].metric("Debt service", f"${res['tot_expense_mm']:.0f}mm")
    cushion = res["tot_income_perf_mm"] + res["tot_swap_mm"] - res["tot_expense_mm"]
    k[4].metric("Coverage cushion", f"${cushion:.0f}mm",
                "income over debt service", delta_color="off")

    n_below = sum(1 for r in res["rows"] if r["dscr"] < 1.0)
    if n_below:
        st.warning(f"{n_below} segment(s) fall below 1.00x coverage under "
                   f"**{measure_label}** at **{mult:.2f}×** stress.")
    else:
        st.success(f"All segments hold ≥1.00x coverage under **{measure_label}** "
                   f"at **{mult:.2f}×** stress; portfolio DSCR **{agg:.2f}x**.")

    # ---- charts -------------------------------------------------------------
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.plotly_chart(dscr_segment_bar(res), width='stretch')
    with cc2:
        st.plotly_chart(dscr_measures_bar(cfg, include_swap, mult), width='stretch')

    # ---- detail table -------------------------------------------------------
    st.subheader("Per-segment DSCR detail")
    tbl = [{
        "Segment": r["name"],
        "Net yield": f"{r['net_yield']*100:.2f}%",
        "Cost of funding": f"{r['cost_of_funding']*100:.2f}%",
        "Assets $mm": f"{r['assets_mm']:.0f}",
        "Debt $mm": f"{r['debt_mm']:.0f}",
        f"Delinq ({measure_label})": f"{r['delinquency']*100:.1f}%",
        "Income (perf) $mm": f"{r['income_perf_mm']:.1f}",
        "Debt service $mm": f"{r['expense_mm']:.1f}",
        "DSCR": f"{r['dscr']:.2f}x",
        "Baseline DSCR": f"{r['base_dscr']:.2f}x",
    } for r in res["rows"]]
    st.dataframe(tbl, width='stretch', hide_index=True)

    st.caption(
        f"Net interest spread as of {res['as_of_spread']} (p.66); delinquency "
        f"aging as of {res['as_of_aging']} (p.18). Delinquent loans are assumed to "
        "stop paying interest (net of any recovery); debt service is held flat. "
        "Business-purpose segments (SFR, SF/MF transitional) share the disclosed "
        "Business Purpose Loans net yield / cost of funding."
    )


# =============================================================================
# Interest-rate / duration tab body
# =============================================================================
def _render_rate_tab(cfg: dict, price: float) -> None:
    irs = cfg["interest_rate_shock"]
    lo, hi = irs["slider_bps_range"]

    st.markdown(
        "Parallel-rate-shock analysis from the 10-Q **Shock Table**. MFA's "
        "disclosed net-portfolio-value change is exactly a duration + convexity "
        "quadratic in the rate shock, so the curve below reproduces the filing to "
        "the dollar at the reported anchors. The **loan MTM P&L** is computed "
        "separately from the disclosed residential-whole-loan duration/convexity; "
        "**hedges + other** (swaps, securitized & other fixed-rate debt) is the "
        "offsetting balance that brings it to the reported net."
    )

    ctrl1, ctrl2 = st.columns([3, 2])
    with ctrl1:
        sel_bps = st.slider("Parallel rate shock (bps)", int(lo), int(hi), 0, 25)
    with ctrl2:
        use_hedges = st.toggle(
            "Apply hedges (net portfolio)", value=True,
            help="On: net-of-hedge Δ (ties to the filing's dNPV). "
                 "Off: re-mark the loan book only (no swap/debt offset).",
        )

    cur = rate_shock_curve(cfg, np.array([float(sel_bps)]))
    loan_pnl = float(cur["loan_pnl_mm"][0])
    hedge_other = float(cur["hedge_other_mm"][0])
    dnpv = float(cur["dnpv_mm"][0])
    pct_eq = float(cur["pct_equity"][0])
    econ_net = float(cur["econ_bvps_net"][0])
    econ_loans = float(cur["econ_bvps_loans_only"][0])
    base = cur["baseline_econ_bvps"]
    shown_bvps = econ_net if use_hedges else econ_loans
    applied = dnpv if use_hedges else loan_pnl

    # ---- KPI row ------------------------------------------------------------
    k = st.columns(6)
    k[0].metric("Loan MTM P&L", f"${loan_pnl:+,.0f}mm",
                help="Residential whole-loan book only (loan duration/convexity).")
    k[1].metric("Hedges + other", f"${hedge_other:+,.0f}mm",
                help="Swaps + securitized/other fixed-rate debt offset.")
    k[2].metric("Net portfolio Δ", f"${dnpv:+,.0f}mm",
                f"{pct_eq*100:+.2f}% of equity", delta_color="normal")
    k[3].metric("Economic BVPS",
                f"${shown_bvps:.2f}", f"{shown_bvps - base:+.2f} vs ${base:.2f}")
    ptb_r = _price_to_book(price, shown_bvps)
    k[4].metric("Price / BV", f"{ptb_r*100:.0f}%",
                f"px ${price:.2f} / BV ${shown_bvps:.2f}", delta_color="off")
    k[5].metric("Applied Δ (this view)", f"${applied:+,.0f}mm",
                "net of hedges" if use_hedges else "loans only")

    # ---- charts -------------------------------------------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(rate_bvps_curve(cfg, sel_bps, use_hedges), width='stretch')
    with c2:
        st.plotly_chart(rate_waterfall(cfg, sel_bps), width='stretch')

    # ---- model-vs-reported tie-out -----------------------------------------
    st.subheader("Shock Table tie-out (model vs reported)")
    rows = rate_shock_table(cfg)
    out = [{
        "Shock (bps)": f"{r['bps']:+d}",
        "Model ΔNPV $000": f"{r['model_npv_change_thousands']:+,.0f}",
        "Reported ΔNPV $000": f"{r['reported_npv_change_thousands']:+,.0f}",
        "Model % equity": f"{r['model_pct_equity']*100:+.2f}%",
        "Reported % equity": f"{r['reported_pct_equity']*100:+.2f}%",
        "Economic BVPS": f"${r['econ_bvps']:.2f}",
    } for r in rows]
    st.dataframe(out, width='stretch', hide_index=True)


# =============================================================================
# Investment thesis (source: MFA_Financial_Thesis.docx)
# Dollar signs are escaped as "\$" so Streamlit markdown does not parse them as
# LaTeX math. Section numbers match the write-up.
# =============================================================================
_THESIS_MD = r"""
## MFA Financial: Value in Levered Residential Credit (Abridged Thesis)

### 1. Overview

In this paper I will go over an attractive investment opportunity in the
Mortgage Real Estate Investment Trust (mREIT) space: MFA Financial (NYSE: MFA).
MFA is an mREIT with close to a 30-year history and invests across various
mortgage credit instruments. MFA, like other mREITs, employs considerable
leverage to enhance returns and offers an attractive yield. Because of that
leverage, and the complexity of the underlying loan books, these vehicles are
often misunderstood by the public.

It is worth distinguishing between the two broad strategies in the space. Agency
mREITs lever quasi-government-guaranteed mortgages (agency MBS), taking
essentially no credit risk but substantial rate and spread risk. Credit-focused
mREITs primarily hold mortgages outside the agency wrapper (non-QM loans,
business-purpose loans, re-performing loans) which offer high yields but carry
genuine credit risk. MFA falls into the latter camp, with a \$12.9 billion
portfolio spanning non-QM (\$5.5B), single-family rental (\$1.2B), transitional
lending (\$1.1B), legacy re-performing/non-performing loans (\$0.9B), and a
\$3.5B agency MBS book, financed with 6.3x leverage against \$1.8B of equity.

### 2. Thesis

MFA currently trades at roughly 70% of its inherent book value, economic book
value per share was \$13.22 at March 31, 2026, and yields approximately 15–16%.
Credit-focused mREITs typically trade between 75–90% of book in normal times,
and much lower in periods of stress and panic; agency REITs typically trade at
low-double-digit to single-digit premiums. So the question the market is
implicitly asking is: why does MFA deserve a discount steeper than its peer
group's normal range? Is something inherently broken?

My answer is no. The discount is a scar from two genuine traumas, COVID and the
2022 rate shock, but the company that earned those scars no longer exists in its
old form. Management rebuilt the liability structure specifically so that the
mechanism that destroyed book value in 2020 (mark-to-market margin calls forcing
asset sales at the bottom) can no longer operate at scale. The stress test in
Section 6 quantifies this: at the current price, the market is effectively paying
you to underwrite a housing decline on the order of −15% to −20% nationally, a
scenario the balance sheet now survives without forced selling, while collecting
a mid-teens yield as you wait.

### 3. Why the Discount Exists

There are 4 major reasons, to me, why MFA trades at such a steep discount even in
current times:

- **Inherent risk and complexity of levered credit.** A book of non-QM,
  transitional, and legacy NPL exposure financed 6.3x is legitimately harder to
  underwrite than a T-bill. Complexity discounts are real, and most investors
  will not read Note 6 of a 10-Q.
- **Large margin calls and a steep drop in book value following COVID-19.** In
  March 2020, mark-to-market repo lenders marked collateral down violently and
  demanded margin. MFA was forced into forbearance with its lenders and into
  asset sales into a dislocated market. Book value was permanently impaired and
  the market has not forgotten (roughly ~36% decline).
- **The 2021–2023 interest rate back-up and mortgage spread blow-out.** The
  steepest rate-hiking cycle in roughly 50 years took agency spreads from around
  50bps to roughly 190bps, levels last seen in 1986, 1998, 2000, and 2008. Book
  value declined further through this period (roughly 25% decline), albeit at a
  considerably slower pace than during COVID, and the dividend was cut from
  \$0.44 to \$0.35.
- **Strong concerns over housing affordability.** With affordability near
  multi-decade lows, investors reasonably fear that home prices are supported by
  thin air, and MFA's equity is a levered claim on home prices.

Against that backdrop, the discount is easy to understand. In a span of three
years the sector absorbed the two things that are inherently worse for levered
mortgage: a pandemic causing forced asset sales, and the steepest rate increase
in half a century. It makes sense that book value fell roughly 56% over a
four-year period. But the market is pricing in the residual pain of the book
value loss, not the current balance sheet.

### 4. The Financing Transformation

The single most important post-COVID change is that management understood the
lesson of March 2020: mark-to-market funding is the kill switch on a levered
credit book. Beginning in 2021, MFA aggressively termed out its liabilities
through securitization and non-mark-to-market facilities; non-MTM funding grew
from roughly \$900 million to \$3.4 billion by Q4 2021, and the securitization
program has continued to compound since. More or less, they shifted their
liability structure completely.

The result, as of March 31, 2026, is a liability stack of roughly \$11.3 billion
that breaks down as follows: \$6.27 billion (56%) is securitized debt
(non-recourse, term, no margin calls) where losses beyond MFA's retained equity
in each deal are borne by securitization bondholders, not by MFA; \$0.27 billion
is non-mark-to-market facilities and senior notes (also immune to margin calls);
and \$4.57 billion (40%) is mark-to-market financing. Critically, \$3.1 billion
of that mark-to-market debt finances the agency MBS book, one of the deepest,
most liquid collateral markets in the world, with modest haircuts and no
credit-driven marks. The mark-to-market debt actually secured by credit-sensitive
whole loans is only about \$1.5 billion, roughly 13% of total financing. That
financing is dominated by transitional loans which are shorter in duration.

This is the structural difference between MFA in 2020 and MFA today. In 2020, a
decline in loan values transmitted directly into margin calls into forced sales
into realized losses. Today, a decline in loan values on 87% of the funding stack
transmits into... nothing immediate. Losses on securitized collateral accrue
against MFA's retained equity slice over time, or past a point, against
bondholders. There is no mechanism for a lender to force MFA to sell whole loans
at the bottom of a panic on that portion of the book.

### 5. The Portfolio Today

The asset side also does not resemble a portfolio priced for distress.
Weighted-average LTV across the \$8.8 billion residential whole loan book is 64%,
meaning the average borrower has 36% equity in front of MFA's position. By
segment: non-QM at 64% LTV with a 739 FICO and just 4.1% 60+ day delinquency;
single-family rental at 66% LTV / 740 FICO / 2.6% delinquent; single-family
transitional at 68% LTV; legacy RPL/NPL at 53% LTV (deep equity cushions, though
a weak 646 FICO and 19% delinquency, these loans were bought as credit-impaired).
The one genuinely hot spot is multifamily transitional: 83% LTV and 16.5% 60+ day
delinquency on a \$407 million book. This is a hotspot for most investors of MFA
as they have taken losses and stopped the program for a while. However there is a
deeper equity cushion here on their leverage and makes up a small portion of
their credit book (~5%).

### 6. Stress Test: What Does It Take to Break This?

Skepticism about levered credit deserves numbers. I built a Monte Carlo model
that stresses home prices and home price dispersion and traces the results
through defaults, loss severities, the financing waterfall, and finally into book
value per share. The mechanics: each segment's loans are simulated with a
loan-level LTV distribution calibrated to the reported weighted averages (using
the WA alone would understate tail losses badly); each path draws a national home
price shock with 20% loan-level dispersion; any loan pushed into negative equity
is assumed to eventually default, with loss severity equal to the equity shortfall
plus 10% foreclosure costs; loans already 60+ days delinquent default regardless
of equity. Losses on securitized collateral are capped at MFA's retained equity
in those deals (non-recourse). Margin calls apply only to mark-to-market debt and
are tested against ~\$270 million of unrestricted cash and unencumbered
securities. Two default assumptions are shown: a draconian "cliff" (100% of
underwater loans default) and a more realistic ramp (default probability rising
from 0% at 100 LTV to 100% at 130 LTV).

**Stressed Economic Book Value per Share (reported: \$13.22)**

[[IMAGE1]]

In a case of armageddon for the housing market, MFA does poorly. This is what we
see in the top right corner. The GFC was roughly a peak-to-trough decline of
~25-30% based on the index and some variance around which housing market. This
also took 6 years to play out, not overnight as this model assumes. Today, given
the persistent housing shortage, I think a more realistic "armageddon" would be
home prices down 10-20% with a variance of 10-20%. In these scenarios MFA is
roughly underwritten to today's prices!

Now, suppose we saw another 200bps rate shock? MFA would again be roughly
underwritten to today's prices.

[[IMAGE2]]

### 7. Return Framework

The return does not require heroics. At a 15–16% yield, the dividend alone
delivers an equity-like return if book value merely holds. If housing muddles
through and MFA simply re-rates to the middle of the normal credit-mREIT range,
say 80–85% of book, that is 15–20% of price appreciation layered on top. The
downside case, per Section 6, is that a −10% to −15% housing decline is roughly
what you are already paying for. The asymmetry comes from being paid mid-teens to
hold an instrument whose worst historical failure mode has been structurally
disabled for ~87% of its funding.

### 8. Risks

The risks are real and worth stating plainly. A severe housing recession (−25% or
worse nationally) impairs book value dramatically regardless of financing
structure; non-recourse debt caps losses, it does not eliminate them. The
multifamily transitional book (83% LTV, 30% delinquent) could produce outsized
losses relative to its size. The \$3.1B agency repo book carries spread and margin
risk even without credit risk; a 2022-style spread blow-out would again pressure
book value and liquidity. Dividend sustainability depends on net interest spread,
and a cut (even a prudent one) would likely hit the stock. Finally, the discount
itself can persist indefinitely; the yield is the compensation for that patience.

### 9. Conclusion

MFA trades like the company that took margin calls in March 2020. It is financed
like a company that never intends to take them again. Fifty-six percent of the
liability stack is non-recourse securitized debt, only ~13% of funding is
mark-to-market against credit-sensitive collateral, borrower equity cushions
average 36%, and a Monte Carlo stress of the actual portfolio shows book value
surviving a GFC-scale housing decline without forced selling. At 70% of a \$13.22
economic book value and a 15–16% yield, the market is paying investors handsomely
to underwrite a risk that has, in large part, already been engineered out of the
structure.
"""


def _small_chart(fig: go.Figure, caption: str, height: int = 320) -> None:
    """Render a compact chart in a narrow column so it doesn't span full width."""
    fig.update_layout(height=height)
    col, _ = st.columns([3, 2])          # chart occupies ~60% of the tab width
    with col:
        st.plotly_chart(fig, width='stretch')
        st.caption(caption)


def _render_thesis_tab(cfg: dict, shocks, disps, med_z: np.ndarray) -> None:
    # Render the thesis, dropping the two LIVE charts in at their markers.
    def _draw(marker: str) -> None:
        if marker == "[[IMAGE1]]":
            _small_chart(
                heatmap_fig(med_z, shocks, disps,
                            "Median Economic BVPS ($/share)", "RdYlGn", ".2f"),
                "Live: stressed median Economic BVPS across home-price shock "
                "(columns) and loan-level dispersion σ (rows).")
        elif marker == "[[IMAGE2]]":
            _small_chart(
                rate_waterfall(cfg, 200),
                "Live: Economic BVPS bridge under a +200bps parallel rate shock.")

    remaining = _THESIS_MD
    for marker in ("[[IMAGE1]]", "[[IMAGE2]]"):
        before, sep, after = remaining.partition(marker)
        st.markdown(before)
        if sep:
            _draw(marker)
        remaining = after
    st.markdown(remaining)
    st.caption("Charts are live: they recompute from config/2026Q1.yaml "
               "(10-Q, quarter ended March 31, 2026) and the sidebar model "
               "settings. Source narrative: MFA_Financial_Thesis.docx.")


# =============================================================================
# Main app
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="MFA Book-Value Model", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(
        f"<style>.stApp{{background:{BG}}} .block-container{{padding-top:1.2rem}}"
        f".stApp, .stApp p, .stApp label {{color:{TEXT}}}</style>",
        unsafe_allow_html=True,
    )

    cfg_path = _parse_config_path()
    if not os.path.exists(cfg_path):
        st.error(f"Config not found: `{cfg_path}`")
        st.stop()
    mtime = os.path.getmtime(cfg_path)

    try:
        cfg = _load_config(cfg_path, mtime)
    except ConfigValidationError as exc:
        st.error("Config failed validation, fix before running:")
        st.code(str(exc))
        st.stop()

    meta = cfg["meta"]
    d0 = cfg["simulation_defaults"]
    bvr = cfg["book_value_reconciliation"]

    st.title("MFA Financial: Book Value Model")
    st.caption(
        f"{meta['entity']} ({meta['ticker']}) · {meta['filing']} · "
        f"as of {meta['as_of_date']} · reported Economic BVPS "
        f"**\\${bvr['reported_economic_bvps']:.2f}**"
    )

    # ---- sidebar inputs -----------------------------------------------------
    sb = st.sidebar
    sb.header("Scenario")
    # slider works in whole percentage points so it displays real home-price
    # changes ("-30%"); converted to a fraction for the model below.
    hp_shock_pct = sb.slider("Home-price change", -50, 0, 0, 1, format="%d%%")
    hp_shock = hp_shock_pct / 100.0
    disp = sb.slider("Loan-level HP dispersion σ", 0.0, 0.30, 0.0, 0.01)
    sys_sigma = sb.slider("Cross-segment systematic σ", 0.0, 0.10, 0.0, 0.005)
    fc = sb.slider("Foreclosure cost (% of post-shock value)", 0.0, 0.30,
                   float(d0["foreclosure_cost"]), 0.01)
    n_sims = int(sb.selectbox("Simulations", [1000, 10000, 50000],
                              index=[1000, 10000, 50000].index(int(d0["n_sims"]))))
    sb.header("Financing / liquidity")
    agency_shock_pct = sb.slider("Agency MBS price change (margin only)", -10.0, 0.0,
                                 float(d0["agency_price_shock"] * 100), 0.5, format="%.1f%%")
    agency_shock = agency_shock_pct / 100.0
    mark_sev = sb.slider("Lender mark severity", 0.5, 1.5,
                         float(d0["lender_mark_severity"]), 0.05)
    fire = sb.slider("Fire-sale discount (forced delever)", 0.0, 0.20,
                     float(d0["fire_sale_discount"]), 0.01)
    haircut = sb.slider("Unencumbered securities haircut", 0.0, 0.40,
                        float(d0["liquidity_haircut"]), 0.01)
    sb.header("Advanced")
    ltv_std = sb.slider("LTV std dev (loan tape)", 0.05, 0.25,
                        float(d0["ltv_std_dev"]), 0.01)
    default_mode = sb.radio("Default trigger", ["negative_equity", "ramp"],
                            index=0 if d0["default_mode"] == "negative_equity" else 1,
                            help="negative_equity: default when post-shock LTV>100%. "
                                 "ramp: P(default) 0%→100% over 100→130 LTV.")
    remark = sb.checkbox("Re-mark FV adjustments to zero (conservative Economic BV)",
                         value=bool(d0["remark_fv_to_zero"]))
    seed = int(sb.number_input("Seed", value=int(d0["seed"]), step=1))
    sb.header("Market")
    mkt = cfg.get("market", {})
    price = float(sb.number_input(
        "Current share price ($)", min_value=0.0,
        value=float(mkt.get("price_per_share", 9.48)), step=0.01,
        help="NYSE:MFA last price, drives the premium/discount-to-book readout.",
    ))
    if mkt.get("price_as_of"):
        sb.caption(f"price as of {mkt['price_as_of']}")

    params_dict = dict(
        hp_shock=hp_shock, loan_dispersion_sigma=disp, systematic_sigma=sys_sigma,
        foreclosure_cost=fc, n_sims=n_sims, agency_price_shock=agency_shock,
        lender_mark_severity=mark_sev, fire_sale_discount=fire,
        liquidity_haircut=haircut, ltv_std_dev=ltv_std,
        loans_per_segment=int(d0["loans_per_segment"]), seed=seed,
        default_mode=default_mode, ramp_ltv_low=float(d0["ramp_ltv_low"]),
        ramp_ltv_high=float(d0["ramp_ltv_high"]), remark_fv_to_zero=remark,
        advance_rate=float(d0.get("advance_rate", 0.90)),
        hpa_cap_up=0.20, hpa_cap_down=-0.90,
    )
    pkey = tuple(sorted(params_dict.items()))
    d = _run(cfg_path, mtime, pkey)
    d["fv_add_baseline"] = (bvr["fv_adj_residential_loans_carrying_mm"]
                            + bvr["fv_adj_securitized_debt_carrying_mm"])
    s = d["summary"]

    base_e = bvr["reported_economic_bvps"]

    # Sensitivity heatmap grid: computed once and shared by the Thesis and
    # credit tabs (the same figure the write-up screenshots showed).
    shocks = tuple(np.round(np.linspace(0.0, -0.25, 6), 3))
    disps = tuple(np.round(np.linspace(0.05, 0.30, 6), 3))
    hkey = tuple(sorted({**params_dict, "n_sims": 1000}.items()))
    med_z, pbr_z = _heatmap(cfg_path, mtime, hkey, shocks, disps, 1000)

    tab_thesis, tab_credit, tab_rate, tab_dscr = st.tabs(
        ["Thesis", "Home-price / credit stress", "Interest-rate / duration",
         "DSCR / delinquency"]
    )

    with tab_thesis:
        _render_thesis_tab(cfg, shocks, disps, med_z)

    # =====================================================================
    # TAB 1: home-price / credit stress (Monte Carlo)
    # =====================================================================
    with tab_credit:
        # ---- KPI row --------------------------------------------------------
        k = st.columns(7)
        delta_e = s["median_econ_bvps"] - base_e
        k[0].metric("Median Economic BVPS", f"${s['median_econ_bvps']:.2f}",
                    f"{delta_e:+.2f} vs ${base_e:.2f}")
        ptb_c = _price_to_book(price, s["median_econ_bvps"])
        k[1].metric("Price / BV", f"{ptb_c*100:.0f}%",
                    f"px ${price:.2f} / BV ${s['median_econ_bvps']:.2f}",
                    delta_color="off")
        k[2].metric("P5 / P95 Econ BVPS",
                    f"${s['p5_econ_bvps']:.2f} / ${s['p95_econ_bvps']:.2f}")
        k[3].metric("Expected loss",
                    f"${s['expected_loss_mm']:.0f}mm",
                    f"{s['expected_loss_pct_equity']*100:.1f}% of equity",
                    delta_color="inverse")
        k[4].metric("P(liquidity breach)", f"{s['p_breach']*100:.1f}%")
        k[5].metric("P(common wiped out)", f"{s['p_common_wiped']*100:.2f}%")
        k[6].metric("P(preferred impaired)", f"{s['p_pref_impaired']*100:.2f}%")

        # ---- narrative ------------------------------------------------------
        mf_dr = next((r["default_rate"] for r in d["seg_table"]
                      if r["key"] == "mf_transitional"), 0.0)
        drop_pct = (base_e - s["median_econ_bvps"]) / base_e * 100
        # NOTE: escape every "$" as "\$": Streamlit markdown treats a bare
        # "$...$" as LaTeX math and would garble the dollar figures.
        st.info(
            f"**Scenario narrative.** At a **{hp_shock:.0%}** home-price change with "
            f"**{disp:.0%}** loan dispersion, median Economic BV falls from "
            f"**\\${base_e:.2f}** to **\\${s['median_econ_bvps']:.2f}** "
            f"(**−{drop_pct:.0f}%**). Expected loss to MFA is "
            f"**\\${s['expected_loss_mm']:.0f}mm** ({s['expected_loss_pct_equity']*100:.0f}% "
            f"of equity). MF-transitional defaults at **{mf_dr:.0%}**. Median margin call "
            f"**\\${s['median_margin_call']:.0f}mm** vs **\\${s['available_liquidity']:.0f}mm** "
            f"available liquidity → **{s['p_breach']*100:.0f}%** of paths breach. "
            f"Common is wiped in **{s['p_common_wiped']*100:.1f}%** of paths."
        )

        # ---- charts row 1 ---------------------------------------------------
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(bvps_hist(d["econ_bvps"], base_e, "Economic BVPS"),
                            width='stretch')
        with c2:
            st.plotly_chart(bridge_waterfall(d, remark), width='stretch')

        # ---- charts row 2 ---------------------------------------------------
        c3, c4 = st.columns([1, 1])
        with c3:
            st.plotly_chart(exp_loss_bars(d), width='stretch')
            st.caption("Expected loss to MFA vs net equity allocated: bars exceeding "
                       "equity indicate segment wipeout.")
        with c4:
            st.plotly_chart(margin_liquidity(d), width='stretch')
            st.caption("Margin-call distribution vs available liquidity (restricted cash "
                       "excluded). Shaded region = liquidity breach.")

        # ---- LTV densities --------------------------------------------------
        st.plotly_chart(ltv_densities(d), width='stretch')
        st.caption("Pre- vs post-shock LTV by segment (post = median systematic path). "
                   "Mass right of the 100% line is negative equity.")

        # ---- heatmap (grid computed above, shared with the Thesis tab) ------
        st.subheader("Sensitivity: HP shock × dispersion")
        h1, h2 = st.tabs(["Median Economic BVPS", "P(liquidity breach)"])
        with h1:
            st.plotly_chart(heatmap_fig(med_z, shocks, disps,
                            "Median Economic BVPS ($/share)", "RdYlGn", ".2f"),
                            width='stretch')
        with h2:
            st.plotly_chart(heatmap_fig(pbr_z * 100, shocks, disps,
                            "P(liquidity breach) %", "Reds", ".0f"),
                            width='stretch')

        # ---- segment table --------------------------------------------------
        st.subheader("Per-segment detail (path-averaged)")
        tbl = [{
            "Segment": r["segment"],
            "Default rate": f"{r['default_rate']*100:.1f}%",
            "Severity": f"{r['severity']*100:.1f}%",
            "Gross loss $mm": f"{r['gross_loss_mm']:.0f}",
            "Loss to MFA $mm": f"{r['loss_to_mfa_mm']:.0f}",
            "To bondholders $mm": f"{r['loss_to_bondholders_mm']:.0f}",
            "Margin call $mm": f"{r['margin_call_mm']:.0f}",
            "Net equity $mm": f"{r['net_equity_mm']:.0f}",
            "Equity remaining $mm": f"{r['equity_remaining_mm']:.0f}",
        } for r in d["seg_table"]]
        st.dataframe(tbl, width='stretch', hide_index=True)

    # =====================================================================
    # TAB 2: interest-rate / duration (Shock Table)
    # =====================================================================
    with tab_rate:
        _render_rate_tab(cfg, price)

    # =====================================================================
    # TAB 3: DSCR / delinquency stress (p.66 net interest spread + p.18 aging)
    # =====================================================================
    with tab_dscr:
        _render_dscr_tab(cfg)

    st.caption("Educational stress model, not investment advice. All inputs from "
               f"MFA's {meta['as_of_date']} 10-Q. Shares held constant (no "
               "buybacks/issuance mid-stress).")


if __name__ == "__main__":
    main()
