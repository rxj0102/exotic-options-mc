"""Streamlit dashboard for the exotic-options-mc engine.

Run with::

    streamlit run app.py

The app exposes seven tabs covering path simulation, exotic pricing,
variance-reduction benchmarking, convergence analysis, risk analytics,
structured products and model documentation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from benchmarks.analytical import bsm_price
from diagnostics.convergence import ConvergenceAnalyzer
from diagnostics.path_visualizer import PathVisualizer
from pricing.asian import AsianOptionPricer
from pricing.barrier import BarrierOptionPricer
from pricing.digital import DigitalOptionPricer
from pricing.lookback import LookbackOptionPricer
from risk.greeks_mc import MCGreeks
from risk.scenario_engine import ExoticScenarioEngine
from risk.var_cvar import ExoticVaR
from simulation.gbm import GBMSimulator
from simulation.heston import HestonSimulator
from simulation.jump_diffusion import MertonJumpDiffusion
from structured_products.autocallable import AutocallablePricer
from structured_products.shark_fin import SharkFinPricer
from utils import plotting
from variance_reduction.combined import CombinedVR, Contract

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Exotic Options MC", layout="wide", initial_sidebar_state="expanded")

# Dark Plotly theme is applied inside the plotting helpers.
st.markdown(
    """
    <style>
    .stMetric { background: #1c1f26; border-radius: 8px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _gbm_paths(S0, T, r, q, sigma, n_paths, n_steps, seed):
    return GBMSimulator(seed=seed).generate_paths(S0, T, r, q, sigma, n_paths, n_steps)


def _badge(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


st.title("📈 Exotic Options — Monte Carlo Engine")
st.caption(
    "Institutional-grade Monte Carlo pricing for path-dependent exotics with a "
    "full variance-reduction suite."
)

tabs = st.tabs(
    [
        "Path Simulator",
        "Exotic Pricer",
        "Variance Reduction Lab",
        "Convergence Analysis",
        "Risk Dashboard",
        "Structured Products",
        "Model Documentation",
    ]
)

# ----------------------------------------------------------------------------
# Sidebar — global market inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Market Inputs")
    S0 = st.number_input("Spot S₀", value=100.0, min_value=1.0)
    r = st.number_input("Risk-free rate r", value=0.05, step=0.01, format="%.4f")
    q = st.number_input("Dividend yield q", value=0.0, step=0.01, format="%.4f")
    sigma = st.number_input("Volatility σ", value=0.20, min_value=0.001, step=0.01, format="%.4f")
    T = st.number_input("Maturity T (years)", value=1.0, min_value=0.01, step=0.25)
    seed = st.number_input("Random seed", value=42, step=1)
    n_paths = st.select_slider(
        "Paths", options=[5_000, 10_000, 50_000, 100_000, 200_000], value=50_000
    )
    n_steps = st.select_slider("Steps", options=[50, 100, 252, 504], value=252)


# ----------------------------------------------------------------------------
# Tab 1 — Path Simulator
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Path Simulator")
    model = st.selectbox("Model", ["GBM", "Heston", "Jump-Diffusion"])
    extra = {}
    if model == "Heston":
        c1, c2, c3, c4, c5 = st.columns(5)
        extra["V0"] = c1.number_input("V₀", value=0.04, format="%.4f")
        extra["kappa"] = c2.number_input("κ", value=2.0)
        extra["theta"] = c3.number_input("θ", value=0.04, format="%.4f")
        extra["xi"] = c4.number_input("ξ", value=0.3)
        extra["rho"] = c5.number_input("ρ", value=-0.7, min_value=-1.0, max_value=1.0)
    elif model == "Jump-Diffusion":
        c1, c2, c3 = st.columns(3)
        extra["lambda_"] = c1.number_input("λ (jump intensity)", value=1.0)
        extra["mu_J"] = c2.number_input("μ_J", value=-0.1)
        extra["sigma_J"] = c3.number_input("σ_J", value=0.15)

    if st.button("Simulate", key="sim_paths"):
        if model == "GBM":
            paths = GBMSimulator(seed=seed).generate_paths(S0, T, r, q, sigma, n_paths, n_steps)
        elif model == "Heston":
            paths, _ = HestonSimulator(seed=seed).generate_paths(
                S0, T, r, q, extra["V0"], extra["kappa"], extra["theta"], extra["xi"], extra["rho"],
                n_paths, n_steps,
            )
        else:
            paths = MertonJumpDiffusion(seed=seed).generate_paths(
                S0, T, r, q, sigma, extra["lambda_"], extra["mu_J"], extra["sigma_J"], n_paths, n_steps,
            )

        pv = PathVisualizer()
        fan = pv.fan_chart_data(paths)
        sample = pv.path_sample_data(paths, 50)
        st.plotly_chart(plotting.fan_chart(fan, sample, f"{model} Path Fan Chart"), use_container_width=True)

        stats = pv.path_statistics(paths)
        st.dataframe(pd.DataFrame([stats]), use_container_width=True)

        hist = pv.terminal_histogram_data(paths)
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_bar(x=hist["bin_centers"], y=hist["density"], name="terminal", marker_color="#50a0ff")
        fig.add_scatter(x=hist["fit_x"], y=hist["fit_y"], name="lognormal fit", line=dict(color="orange"))
        fig.update_layout(template="plotly_dark", title="Terminal distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download paths (CSV)",
            data=pd.DataFrame(paths).to_csv(index=False).encode(),
            file_name="paths.csv",
        )


# ----------------------------------------------------------------------------
# Tab 2 — Exotic Pricer
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Exotic Pricer")
    family = st.selectbox("Option family", ["Asian", "Lookback", "Barrier", "Digital"])
    K = st.number_input("Strike K", value=100.0)
    paths = _gbm_paths(S0, T, r, q, sigma, n_paths, n_steps, seed)

    result = None
    analytic = None
    if family == "Asian":
        freq = st.selectbox("Averaging frequency", ["continuous", "daily", "weekly", "monthly"])
        floating = st.checkbox("Floating strike")
        otype = st.radio("Type", ["call", "put"], horizontal=True)
        ap = AsianOptionPricer(freq)
        result = ap.price_arithmetic(paths, K, r, T, otype, floating)
        if not floating:
            analytic = ap.price_geometric_analytical(S0, K, T, r, q, sigma, otype)
            st.caption("Analytical shown is the *geometric* Asian (lower bound for arithmetic).")
    elif family == "Lookback":
        floating = st.checkbox("Floating strike", value=True)
        otype = st.radio("Type", ["call", "put"], horizontal=True)
        lb = LookbackOptionPricer()
        if floating:
            result = lb.price_floating_strike(paths, r, T, otype, sigma=sigma, continuity_correction=True)
            analytic = (
                lb.floating_call_analytical(S0, T, r, q, sigma)
                if otype == "call"
                else lb.floating_put_analytical(S0, T, r, q, sigma)
            )
        else:
            result = lb.price_fixed_strike(paths, K, r, T, otype, sigma=sigma, continuity_correction=True)
    elif family == "Barrier":
        bt = st.selectbox("Barrier type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"])
        otype = st.radio("Type", ["call", "put"], horizontal=True)
        H = st.number_input("Barrier H", value=120.0)
        rebate = st.number_input("Rebate", value=0.0)
        bp = BarrierOptionPricer()
        result = bp.price_mc(paths, K, H, r, T, bt, otype, rebate=rebate, sigma=sigma, brownian_bridge=True)
        analytic = bp.price_analytical(S0, K, H, T, r, q, sigma, bt, otype, rebate)
    else:  # Digital
        sub = st.selectbox("Subtype", ["cash_or_nothing", "asset_or_nothing", "one_touch", "no_touch"])
        otype = st.radio("Type", ["call", "put"], horizontal=True)
        H = st.number_input("Barrier H (touch only)", value=120.0)
        dp = DigitalOptionPricer()
        result = dp.price_mc(paths, K, r, T, sub, otype, H=H)
        analytic = dp.price_analytical(S0, K, T, r, q, sigma, sub, otype, H=H)

    if result is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("MC Price", f"{result.price:.4f}", f"± {result.ci_width / 2:.4f}")
        if analytic is not None:
            dev_bps = (result.price - analytic) / analytic * 1e4 if analytic else 0.0
            c2.metric("Analytical", f"{analytic:.4f}", f"{dev_bps:.1f} bps")
            within = abs(result.price - analytic) < 3 * result.stderr + 0.05
            c3.metric("Validation", _badge(within))
        st.caption(str(result))

        # Payoff diagram at expiry.
        ST = np.linspace(S0 * 0.4, S0 * 1.8, 200)
        if family in ("Asian",) or (family == "Lookback"):
            payoff = np.maximum(ST - K, 0.0) if otype == "call" else np.maximum(K - ST, 0.0)
        elif family == "Barrier":
            payoff = np.maximum(ST - K, 0.0) if otype == "call" else np.maximum(K - ST, 0.0)
        else:
            payoff = (ST > K).astype(float) if otype == "call" else (ST < K).astype(float)
        st.plotly_chart(plotting.payoff_diagram(ST, payoff), use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 3 — Variance Reduction Lab
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Variance Reduction Lab")
    st.caption("Benchmarked on an at-the-money arithmetic Asian call.")
    vr_paths = st.select_slider("Paths (per method)", options=[10_000, 20_000, 50_000], value=20_000, key="vr_n")
    if st.button("Run all methods", key="run_vr"):
        contract = Contract(S0=S0, K=K if "K" in dir() else 100.0, T=T, r=r, q=q, sigma=sigma)
        table = CombinedVR(seed=seed).efficiency_table(contract, n_paths=vr_paths)
        st.dataframe(table, use_container_width=True)
        st.plotly_chart(
            plotting.bar_comparison(table["Method"], table["StdErr"], "Standard Error by Method", "StdErr"),
            use_container_width=True,
        )
        best = table.loc[table["VR_Ratio"].idxmax()]
        st.success(f"**{best['Method']}** delivers a {best['VR_Ratio']:.0f}× variance reduction vs crude MC.")


# ----------------------------------------------------------------------------
# Tab 4 — Convergence Analysis
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Convergence Analysis")
    if st.button("Run convergence", key="run_conv"):
        ap = AsianOptionPricer("continuous")
        g = GBMSimulator(seed=seed)

        def pricer(n):
            p = g.generate_paths(S0, T, r, q, sigma, int(n), 100)
            return ap.price_arithmetic(p, 100.0, r, T, "call")

        ca = ConvergenceAnalyzer()
        df = ca.price_vs_paths(pricer, [1_000, 5_000, 10_000, 50_000, 100_000, 500_000])
        st.plotly_chart(plotting.convergence_plot(df, title="Price vs n_paths"), use_container_width=True)
        rate = ca.convergence_rate(df)
        st.metric("Fitted convergence rate", f"{rate:.3f}", "theory ≈ -0.5")
        prec = pd.DataFrame(
            {
                "Target StdErr": [0.10, 0.05, 0.01],
                "Paths needed": [ca.path_count_for_precision(pricer, t, 10_000) for t in (0.10, 0.05, 0.01)],
            }
        )
        st.dataframe(prec, use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 5 — Risk Dashboard
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Risk Dashboard")
    K_risk = st.number_input("Strike (vanilla call proxy)", value=100.0, key="k_risk")
    horizon = st.slider("VaR horizon (days)", 1, 20, 1)
    if st.button("Compute risk", key="run_risk"):
        value_fn = lambda s: bsm_price(s, K_risk, T, r, q, sigma, "call")
        ev = ExoticVaR(value_fn, seed=seed)
        c1, c2 = st.columns(2)
        c1.metric("99% VaR", f"{ev.var(S0, sigma, r, 0.99, horizon, 20_000):.4f}")
        c2.metric("99% CVaR", f"{ev.cvar(S0, sigma, r, 0.99, horizon, 20_000):.4f}")

        pnl = ev.pnl_distribution_data(S0, sigma, r, horizon, 20_000)
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(x=pnl["bin_centers"], y=pnl["counts"], marker_color="#50a0ff"))
        fig.add_vline(x=-pnl["var"], line_dash="dash", line_color="orange", annotation_text="VaR")
        fig.add_vline(x=-pnl["cvar"], line_dash="dot", line_color="red", annotation_text="CVaR")
        fig.update_layout(template="plotly_dark", title="P&L distribution")
        st.plotly_chart(fig, use_container_width=True)

        mg = MCGreeks(seed=seed, n_steps=1)
        payoff_fn = lambda p: np.maximum(p[:, -1] - K_risk, 0.0)
        gs = mg.all_greeks(S0, K_risk, T, r, q, sigma, payoff_fn, n_paths=100_000)
        st.dataframe(
            pd.DataFrame(
                [{"Delta": gs.delta, "Gamma": gs.gamma, "Vega": gs.vega, "Theta": gs.theta, "Rho": gs.rho, "Method": gs.method}]
            ),
            use_container_width=True,
        )

        eng = ExoticScenarioEngine(
            lambda S0, sigma, r, T: bsm_price(S0, K_risk, T, r, q, sigma, "call"),
            {"S0": S0, "sigma": sigma, "r": r, "T": T},
        )
        st.dataframe(pd.DataFrame(eng.run_all()), use_container_width=True)


# ----------------------------------------------------------------------------
# Tab 6 — Structured Products
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Structured Products")
    product = st.selectbox("Product", ["Autocallable", "Shark Fin"])
    if product == "Autocallable":
        c1, c2, c3 = st.columns(3)
        ac_barrier = c1.number_input("Autocall barrier (×S₀)", value=1.0)
        coupon = c2.number_input("Coupon (p.a.)", value=0.08)
        cap_barrier = c3.number_input("Capital barrier (×S₀)", value=0.6)
        tenor = st.slider("Tenor (years)", 1, 5, 3)
        if st.button("Price autocallable", key="run_ac"):
            res = AutocallablePricer(seed=seed).price(
                S0, ac_barrier, coupon, cap_barrier, float(tenor),
                list(range(1, tenor + 1)), r, q, sigma, n_paths=50_000,
            )
            st.metric("Fair value", f"{res.price:.3f}")
            st.metric("Expected life (yrs)", f"{res.extra['expected_life']:.2f}")
            probs = pd.DataFrame({"Obs date (yr)": res.extra["obs_dates"], "Autocall prob": res.extra["autocall_probs"]})
            st.plotly_chart(
                plotting.bar_comparison([str(d) for d in probs["Obs date (yr)"]], probs["Autocall prob"], "Autocall probability", "P"),
                use_container_width=True,
            )
    else:
        c1, c2, c3 = st.columns(3)
        barrier = c1.number_input("Knock-out barrier", value=130.0)
        rebate = c2.number_input("Rebate", value=5.0)
        Ksf = c3.number_input("Strike", value=100.0, key="ksf")
        if st.button("Price shark fin", key="run_sf"):
            res = SharkFinPricer(seed=seed).price(S0, Ksf, barrier, rebate, T, r, q, sigma, n_paths=50_000)
            st.metric("Fair value", f"{res.price:.3f}")
            comp = res.extra["components"]
            import plotly.graph_objects as go

            fig = go.Figure(go.Pie(labels=list(comp.keys()), values=list(comp.values())))
            fig.update_layout(template="plotly_dark", title="Component breakdown")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Knock-out probability: {res.extra['knockout_probability']:.1%}")


# ----------------------------------------------------------------------------
# Tab 7 — Model Documentation
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Model Documentation")
    st.markdown("### Path dependency")
    st.markdown(
        "Asian, lookback and barrier options depend on the *entire* price path, "
        "not just the terminal value, which is why closed forms are scarce and "
        "Monte Carlo is the workhorse."
    )
    st.markdown("### Variance-reduction estimators")
    st.latex(r"\hat P_{\text{anti}} = \tfrac12\big(f(Z) + f(-Z)\big)")
    st.latex(r"\hat P_{\text{cv}} = \bar X - \beta\,(\bar C - \mathbb{E}[C]),\quad \beta = \frac{\operatorname{Cov}(X,C)}{\operatorname{Var}(C)}")
    st.latex(r"\hat P_{\text{IS}} = \mathbb{E}^{\tilde P}\!\big[f(S)\,e^{-\theta W_T - \tfrac12\theta^2 T}\big]")
    st.markdown("### Convergence theory")
    st.latex(r"\text{Crude MC: } O(N^{-1/2}) \qquad \text{QMC: } O\!\big((\log N)^d / N\big)")
    st.markdown("### Model comparison")
    st.dataframe(
        pd.DataFrame(
            {
                "Model": ["GBM", "Heston", "Merton Jump-Diffusion"],
                "Vol dynamics": ["constant", "stochastic (mean-reverting)", "constant + jumps"],
                "Key params": ["σ", "κ, θ, ξ, ρ, V₀", "λ, μ_J, σ_J"],
                "Use case": ["baseline", "vol smile / skew", "crash risk / fat tails"],
            }
        ),
        use_container_width=True,
    )
    st.markdown(
        "**References:** Kemna & Vorst (1990) · Reiner & Rubinstein (1991) · "
        "Goldman, Sosin & Gatto (1979) · Broadie, Glasserman & Kou (1997) · "
        "Glasserman, *Monte Carlo Methods in Financial Engineering* (2003)."
    )
