"""
valuation.py – Skeleton for an FCFF Discounted-Cash-Flow model
--------------------------------------------------------------

Dependencies (install in your virtual environment):
    pip install pandas yfinance fredapi python-dotenv openpyxl

Author: Lucas Duran
"""

from pathlib import Path
import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
import re
from typing import Sequence

# --------------------------------------------------------------------------- #
# 1 · Environment & paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # adjust if needed
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Load API keys from .env at project root
load_dotenv(PROJECT_ROOT / ".env")
FRED_KEY = os.getenv("FRED_API_KEY")       # <- make sure .env contains this


# --------------------------------------------------------------------------- #
# 2 · Raw data pull
# --------------------------------------------------------------------------- #
def get_financials(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Pull the most recent annual & trailing-twelve-month financial statements
    from Yahoo Finance.

    Returns
    -------
    dict
        {
            "income_stmt": DataFrame,
            "balance_sheet": DataFrame,
            "cashflow": DataFrame
        }
    """
    tkr = yf.Ticker(ticker)

    fin_dict = {
        "income_stmt": tkr.income_stmt,
        "balance_sheet": tkr.balance_sheet,
        "cashflow": tkr.cashflow,
    }

    # Optionally dump raw CSVs for inspection
    for name, df in fin_dict.items():
        df.to_csv(DATA_DIR / f"{ticker}_{name}.csv")

    return fin_dict


# --------------------------------------------------------------------------- #
# 3 · Helper calculations
# --------------------------------------------------------------------------- #

# Utility function to pick a row from a DataFrame based on exact labels or keywords.
def pick_row(
    df: pd.DataFrame,
    aliases: Sequence[str] | None = None,
    must_contain: Sequence[str] | None = None,
) -> pd.Series:
    """
    Return the first matching row from `df`.

    Parameters
    ----------
    df : DataFrame whose index holds line-item labels.
    aliases : exact labels to try first (case-sensitive).
    must_contain : list of substrings; row is a hit if **all** appear
                   (case-insensitive) in the label.

    Raises
    ------
    KeyError if nothing matches.
    """
    # 1) exact matches (fast path)
    if aliases:
        for name in aliases:
            if name in df.index:
                return df.loc[name]

    # 2) keyword fallback (robust path)
    if must_contain:
        pattern = re.compile(r".*".join(map(re.escape, must_contain)), re.I)
        for label in df.index:
            if pattern.search(label):
                return df.loc[label]

    # Nothing hit → raise
    raise KeyError(
        f"No row found. Tried aliases={aliases} and keywords={must_contain}"
    )

def calc_historical_fcff(statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute historical Free Cash-Flow to the Firm (FCFF).

    FCFF = NOPAT + Depreciation & Amortization – CapEx – ΔNWC
    """
    income = statements["income_stmt"]
    bal = statements["balance_sheet"]
    cf = statements["cashflow"]

    # -------------- core line items -------------- #
    nopat = income.loc["EBIT"] * (1 - 0.21)          # assume 21 % tax for now
    dep_amort = pick_row(cf, aliases=["Depreciation & Amortization", "Depreciation & amortization",],   # exact match
                          must_contain=["Depreciation"])          # any label containing this
    capex = -pick_row(cf, aliases=["Capital Expenditure", "Capital Expenditures"],    # Yahoo returns negative
                      must_contain=["capital", "expend"]) 

    # Net working-capital components
    nwc = (
        bal.loc["Accounts Receivable"]
        + bal.loc["Inventory"]
        - bal.loc["Accounts Payable"]
    )
    delta_nwc = nwc.diff(periods=-1).fillna(0)  # year-over-year change

    fcff = nopat + dep_amort - capex - delta_nwc
    fcff_df = fcff.to_frame(name="FCFF").T     # nicer shape (years as columns)

    return fcff_df


def get_risk_free_rate() -> float:
    """Fetch latest 10-year Treasury yield from FRED and return as decimal."""
    fred = Fred(api_key=FRED_KEY)
    latest_pct = fred.get_series_latest_release("DGS10").iloc[-1]
    return latest_pct / 100.0


def calc_wacc(
    risk_free: float,
    beta: float,
    mkt_premium: float,
    cost_debt: float,
    tax_rate: float,
    target_debt_pct: float = 0.40,
) -> float:
    """Weighted-Average Cost of Capital (simple version)."""
    ke = risk_free + beta * mkt_premium
    wacc = ke * (1 - target_debt_pct) + cost_debt * target_debt_pct * (1 - tax_rate)
    return wacc


# --------------------------------------------------------------------------- #
# 4 · Projection placeholder
# --------------------------------------------------------------------------- #
def forecast_fcff(
    historical_fcff: pd.DataFrame,
    assumptions: dict,
) -> pd.DataFrame:
    """
    Forecast FCFF for the explicit projection period.

    This is a placeholder; it will be implemented on Day 3.
    """
    raise NotImplementedError("Forecasting routine not yet implemented.")


# --------------------------------------------------------------------------- #
# 5 · Main script hook – quick sanity test
# --------------------------------------------------------------------------- #
def main(ticker: str = "AAPL"):
    print(f"Pulling financials for {ticker} …")
    fin = get_financials(ticker)
    print("Income-statement shape:", fin["income_stmt"].shape)

    fcff_hist = calc_historical_fcff(fin)
    print("\nHistorical FCFF:")
    print(fcff_hist.head())


if __name__ == "__main__":
    main()
