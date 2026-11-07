"""Setup script for the exotic-options-mc package."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (
    (ROOT / "README.md").read_text(encoding="utf-8")
    if (ROOT / "README.md").exists()
    else ""
)

setup(
    name="exotic-options-mc",
    version="0.1.0",
    description=(
        "Institutional-grade Monte Carlo simulation engine for exotic options "
        "with a full variance reduction suite."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="rxj0102",
    url="https://github.com/rxj0102/exotic-options-mc",
    license="MIT",
    packages=find_packages(
        exclude=("tests", "tests.*", "notebooks", "paper", "results")
    ),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "streamlit>=1.30",
        "plotly>=5.18",
        "matplotlib>=3.7",
        "reportlab>=4.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    keywords=[
        "monte-carlo",
        "exotic-options",
        "variance-reduction",
        "quasi-monte-carlo",
        "quantitative-finance",
        "derivatives",
        "structured-products",
    ],
)
