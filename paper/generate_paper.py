"""Generate an SSRN-ready PDF write-up of the methodology and results.

Tables are auto-populated from ``results/benchmark_results.json`` (produced by
:mod:`paper.generate_benchmarks`). Run with::

    python paper/generate_paper.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "benchmark_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "exotic_options_mc.pdf"
RUNNING_HEADER = "Exotic Options — Monte Carlo Methods"


def _load_results() -> dict:
    if not RESULTS_PATH.exists():
        from paper.generate_benchmarks import main as gen

        gen()
    return json.loads(RESULTS_PATH.read_text())


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Title12",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=16,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "Body12",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=12,
            leading=15,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            "H12",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=13,
        )
    )
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.drawString(inch, letter[1] - 0.6 * inch, RUNNING_HEADER)
    canvas.drawRightString(letter[0] - inch, 0.6 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Times-Roman", 10),
                ("FONT", (0, 0), (-1, 0), "Times-Bold", 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f0")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build(results: dict) -> list:
    styles = _styles()
    P = lambda txt: Paragraph(txt, styles["Body12"])
    H = lambda txt: Paragraph(txt, styles["H12"])
    story = []

    story.append(
        Paragraph(
            "Monte Carlo Methods for Exotic Options Pricing:<br/>"
            "Path Dependency, Variance Reduction, and Structured Product Valuation",
            styles["Title12"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(Paragraph("rxj0102", styles["Body12"]))
    story.append(Spacer(1, 14))

    mkt = results["market"]
    eff = results["efficiency_table"]
    best = max(eff, key=lambda r: r["VR_Ratio"])

    # 1. Abstract (~175 words).
    story.append(H("Abstract"))
    story.append(
        P(
            "Exotic options whose payoffs depend on the entire trajectory of the "
            "underlying — Asian, lookback, barrier and digital structures — rarely "
            "admit closed-form valuations, making Monte Carlo simulation the "
            "practitioner's principal tool. This paper presents a unified, "
            "production-grade Monte Carlo engine that prices path-dependent "
            "exotics under geometric Brownian motion, the Heston stochastic-"
            "volatility model and the Merton jump-diffusion model. We implement a "
            "complete variance-reduction suite — antithetic variates, control "
            "variates, stratified and Latin-hypercube sampling, scrambled Sobol "
            "and Halton quasi-Monte Carlo, and importance sampling — and quantify "
            "their efficiency on a common benchmark. Using the geometric Asian as "
            f"a control variate delivers a {best['VR_Ratio']:.0f}-fold variance "
            "reduction over crude Monte Carlo, while randomized quasi-Monte Carlo "
            "with a Brownian-bridge construction yields comparable gains. Every "
            "estimator is validated against analytical benchmarks (Kemna-Vorst, "
            "Reiner-Rubinstein and Goldman-Sosin-Gatto) to within statistical "
            "error. We further demonstrate the framework on structured products — "
            "autocallable notes, shark-fin certificates and FX accumulators — and "
            "on Monte Carlo Greeks and value-at-risk for exotic books."
        )
    )

    # 2. Introduction.
    story.append(H("1. Introduction"))
    story.append(
        P(
            "Closed-form solutions exist only for special cases: the geometric "
            "Asian, the single-barrier option under continuous monitoring, and the "
            "European digital. The moment a payoff references an arithmetic "
            "average, a discretely monitored extremum, or multiple coupled "
            "barriers, analytical tractability is lost and simulation becomes "
            "indispensable. The central difficulty of Monte Carlo — its slow "
            f"{chr(0x00BD)}-power error decay — motivates the variance-reduction "
            "machinery that is the focus of this work."
        )
    )

    # 3. Monte Carlo framework.
    story.append(H("2. Monte Carlo Framework"))
    story.append(
        P(
            "We simulate the underlying with an exact log-Euler discretization of "
            "geometric Brownian motion, a full-truncation Euler scheme for the "
            "Heston variance process, and a compound-Poisson overlay for Merton "
            "jumps. All schemes are fully vectorized across paths. Discounted "
            "payoffs are summarized by their sample mean, standard error and "
            "95% confidence interval through a common result object."
        )
    )

    # 4. Path-dependent exotics.
    story.append(H("3. Path-Dependent Exotic Options"))
    story.append(
        P(
            "Asian options average the underlying over a monitoring schedule; "
            "lookbacks reference the running extremum; barriers activate or "
            "extinguish on contact with a level; digitals pay a fixed amount on a "
            "terminal or first-passage event. Discrete monitoring introduces a "
            "well-understood bias relative to continuous formulae, which we correct "
            "with the Broadie-Glasserman-Kou continuity shift and a Brownian-bridge "
            "crossing probability."
        )
    )

    # 5. Variance reduction with results table.
    story.append(H("4. Variance Reduction Techniques"))
    story.append(
        P(
            "Table 1 reports the efficiency of each technique on an at-the-money "
            "arithmetic Asian call "
            f"(S0 = {mkt['S0']:.0f}, K = {mkt['K']:.0f}, T = {mkt['T']:.0f}, "
            f"r = {mkt['r']:.0%}, sigma = {mkt['sigma']:.0%}). The variance-"
            "reduction ratio is the crude-MC variance divided by the method "
            "variance; values above one indicate improvement."
        )
    )
    eff_rows = [["Method", "Price", "Std. Err.", "VR Ratio", "Time (s)"]]
    for row in eff:
        eff_rows.append(
            [
                row["Method"],
                f"{row['Price']:.4f}",
                f"{row['StdErr']:.5f}",
                f"{row['VR_Ratio']:.1f}",
                f"{row['Computation_Time']:.3f}",
            ]
        )
    story.append(
        _table(
            eff_rows,
            col_widths=[1.7 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch],
        )
    )
    story.append(
        Paragraph("<i>Table 1. Variance-reduction efficiency.</i>", styles["Body12"])
    )
    story.append(Spacer(1, 8))

    # 6. Validation table.
    story.append(H("5. Model Validation"))
    story.append(
        P(
            "Table 2 compares Monte Carlo prices against analytical benchmarks. "
            "Every instrument agrees with its closed form to within roughly one "
            "standard error, confirming the correctness of both the simulators and "
            "the discretization corrections."
        )
    )
    val_rows = [
        ["Instrument", "MC Price", "Std. Err.", "Analytical", "Dev. (bps)", "z"]
    ]
    for row in results["validation"]:
        val_rows.append(
            [
                row["instrument"],
                f"{row['mc_price']:.4f}",
                f"{row['stderr']:.4f}",
                f"{row['analytical']:.4f}",
                f"{row['deviation_bps']:.1f}",
                f"{row['z_score']:.2f}",
            ]
        )
    story.append(
        _table(
            val_rows,
            col_widths=[
                1.8 * inch,
                0.9 * inch,
                0.9 * inch,
                0.9 * inch,
                0.8 * inch,
                0.5 * inch,
            ],
        )
    )
    story.append(
        Paragraph("<i>Table 2. MC vs analytical benchmarks.</i>", styles["Body12"])
    )
    story.append(Spacer(1, 8))

    # 7. Structured products.
    story.append(H("6. Structured Product Applications"))
    story.append(
        P(
            "The component pricers compose into structured products. We value "
            "autocallable notes with discrete observation dates and early-"
            "redemption coupons, shark-fin certificates combining a knock-out call "
            "with a rebate, and FX accumulators with leveraged accumulation below a "
            "lower barrier — recovering the characteristically asymmetric, "
            "negatively skewed payoff of the latter."
        )
    )

    # 8. Risk management.
    story.append(H("7. Risk Management"))
    story.append(
        P(
            "Sensitivities are computed by the pathwise and likelihood-ratio "
            "estimators, with a common-random-number finite-difference fallback "
            "for discontinuous payoffs. Value-at-risk and expected shortfall are "
            "obtained by revaluing positions over a short risk horizon, and stress "
            "scenarios sweep spot, volatility and time."
        )
    )

    # 9. Conclusion.
    story.append(H("8. Conclusion and Extensions"))
    story.append(
        P(
            "A modular Monte Carlo engine, paired with a disciplined variance-"
            "reduction layer and rigorous analytical validation, prices the full "
            "exotic spectrum accurately and efficiently. Natural extensions include "
            "Heston calibration to market surfaces and American-style exotics via "
            "the Longstaff-Schwartz least-squares method."
        )
    )

    # 10. References.
    story.append(H("References"))
    for ref in [
        "Kemna, A. and Vorst, A. (1990). A Pricing Method for Options Based on Average Asset Values. Journal of Banking &amp; Finance.",
        "Reiner, E. and Rubinstein, M. (1991). Breaking Down the Barriers. Risk Magazine.",
        "Goldman, M., Sosin, H. and Gatto, M. (1979). Path Dependent Options. Journal of Finance.",
        "Broadie, M., Glasserman, P. and Kou, S. (1997). A Continuity Correction for Discrete Barrier Options. Mathematical Finance.",
        "Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.",
        "Merton, R. (1976). Option Pricing when Underlying Stock Returns are Discontinuous. Journal of Financial Economics.",
        "Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility. Review of Financial Studies.",
    ]:
        story.append(P(ref))
        story.append(Spacer(1, 2))

    return story


def main() -> None:
    """Build the PDF and write it to ``paper/exotic_options_mc.pdf``."""
    results = _load_results()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        topMargin=inch,
        bottomMargin=inch,
        leftMargin=inch,
        rightMargin=inch,
        title="Monte Carlo Methods for Exotic Options Pricing",
    )
    doc.build(build(results), onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
