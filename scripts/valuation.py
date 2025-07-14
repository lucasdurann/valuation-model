"""
valuation.py – Skeleton for an FCFF Discounted-Cash-Flow model
--------------------------------------------------------------

Dependencies (install in your virtual environment):
    pip install pandas yfinance fredapi python-dotenv openpyxl

Author: Lucas Duran
"""

from pathlib import Path
import os
import sys

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
    Return a dict of cleaned DataFrames:
        income_stmt, balance_sheet, cashflow
    """
    tkr = yf.Ticker(ticker)

    # ---------- helper 1: clean annual tables ----------
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = (df.index.astype(str)
                              .str.strip()
                              .str.lower()
                              .str.replace(r"\s+", " ", regex=True))
        df = df.iloc[:, :5]                       # keep latest 5 cols
        parsed = pd.to_datetime(df.columns, errors="coerce")
        df.columns = [
        str(ts.year) if not pd.isna(ts) else str(orig)
        for orig, ts in zip(df.columns, parsed)
        ]
        return df.astype(float)

    # ---------- fetch three annual statements ----------
    fin = {
        "income_stmt":   _clean(tkr.income_stmt),
        "balance_sheet": _clean(tkr.balance_sheet),
        "cashflow":      _clean(tkr.cashflow),
    }

    # Optional cache for debugging
    for name, df in fin.items():
        (DATA_DIR / f"{ticker}_{name}.csv").write_text(df.to_csv())

    return fin


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

# Tidy up the financial statements into a single DataFrame.
def tidy_statements(fin: dict) -> pd.DataFrame:
    rows = [
        pick_row(fin["income_stmt"],
             aliases=["total revenue", "operating revenue"],
             must_contain="revenue").rename("revenue"),
        pick_row(fin["income_stmt"],
             aliases=["ebit", "operating income"],
             must_contain="operating income|ebit").rename("ebit"),
        pick_row(fin["income_stmt"],
             aliases=["income tax expense", "tax provision"],
             must_contain="tax").rename("tax"),
        pick_row(fin["cashflow"],
             aliases=["depreciation & amortization", "depreciation"],
             must_contain="depreci").rename("dep_amort"),
        pick_row(fin["cashflow"],
             aliases=["capital expenditure", "capital expenditures"],
             must_contain="capital expend").rename("capex"),
        pick_row(fin["balance_sheet"],
             aliases=["total assets"],
             must_contain="total assets").rename("assets"),
        pick_row(fin["balance_sheet"],
             aliases=["total debt"],
             must_contain="total debt").rename("debt"),
        pick_row(fin["balance_sheet"],
             aliases=["cash", "cash and cash equivalents"],
             must_contain="cash").rename("cash"),
        pick_row(fin["balance_sheet"],
             aliases=["accounts receivable"],
             must_contain="accounts rec").rename("ar"),
        pick_row(fin["balance_sheet"],
             aliases=["inventory"],
             must_contain="inventory").rename("inventory"),
        pick_row(fin["balance_sheet"],
             aliases=["accounts payable"],
             must_contain="accounts pay").rename("ap"),
    ]
    tidy = pd.concat(rows, axis=1).T
    # move TTM to right-most column if not already
    if "ttm" in tidy.columns:
        cols = [c for c in tidy.columns if c != "ttm"] + ["ttm"]
        tidy = tidy[cols]
    return tidy


def calc_historical_fcff(statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute historical Free Cash-Flow to the Firm (FCFF).

    FCFF = NOPAT + Depreciation & Amortization – CapEx – ΔNWC
    """
    income = statements["income_stmt"]
    bal = statements["balance_sheet"]
    cf = statements["cashflow"]

    # -------------- core line items -------------- #
    nopat = pick_row(income, aliases=["ebit"]).astype(float) * (1 - 0.21)          # assume 21 % tax for now
    dep_amort = pick_row(cf, aliases=["Depreciation & Amortization", "Depreciation & amortization",],   # exact match
                          must_contain=["Depreciation"])   # any label containing this
    capex = -pick_row(cf, aliases=["Capital Expenditure", "Capital Expenditures"],    # Yahoo returns negative
                      must_contain=["capital", "expend"])

    # Net working-capital components
    nwc = (
        bal.loc["accounts receivable"]
      + bal.loc["inventory"]
      - bal.loc["accounts payable"]
   )
    # compute ΔNWC and *ensure* a zero for "ttm" by reindexing to nopat.index
    delta_nwc = (
               nwc.diff(periods=-1)     # year-over-year change
               .fillna(0)            # fill missing annual change
               .reindex(nopat.index, fill_value=0)
   )

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

def dump_to_excel(ticker: str, tidy: pd.DataFrame):
    xl_path = PROJECT_ROOT / "templates" / "DCF_Model.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as w:
        tidy.to_excel(w,
                      sheet_name="Raw_FS",
                      startrow=0, startcol=0,  # B2
                      header=True, index=True)
    print(f"✅  Raw_FS updated for {ticker} → {xl_path.name}")

# --------------------------------------------------------------------------- #
# 5 · Main script hook – quick sanity test
# --------------------------------------------------------------------------- #
def main(ticker: str = "AAPL"):
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    print(f"Pulling financials for {ticker} …")
    fin = get_financials(ticker)
    tidy = tidy_statements(fin)
    dump_to_excel(ticker, tidy)
    print("Income-statement shape:", fin["income_stmt"].shape)

    fcff_hist = calc_historical_fcff(fin)
    print("\nHistorical FCFF:")
    print(fcff_hist.head())


if __name__ == "__main__":
    main()
