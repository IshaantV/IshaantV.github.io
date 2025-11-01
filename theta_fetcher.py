"""
theta_fetcher.py

Modular script to fetch option expirations/strikes and historical underlying data
from ThetaData, compute rolling volatility, and plot price vs. volatility.

Credentials are read from environment variables:
  - THETA_USER
  - THETA_PASS

Example:
  python theta_fetcher.py --root MSFT --start 2012-06-01 --end 2022-11-14 --interval-ms 60000
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thetadata import ThetaClient
from thetadata.enums import DateRange, DataType, SecType

# ---- Logging configuration ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define your ThetaData credentials for the program to work

# ---- Helper / utility functions ----
def get_env_credentials() -> Dict[str, str]:
    """Read Theta credentials from environment variables and return them.

    Raises:
        RuntimeError: if required variables are missing.
    """
    user = os.getenv("THETA_USER")
    passwd = os.getenv("THETA_PASS")
    if not user or not passwd:
        raise RuntimeError("Please set THETA_USER and THETA_PASS environment variables.")
    return {"username": user, "passwd": passwd}


def weighted_mid_price_row(row: pd.Series) -> float:
    """Compute a weighted mid price from a row using ASK/BID sizes and prices.

    Returns np.nan if inputs are invalid.
    """
    try:
        ask_size = float(row.get(DataType.ASK_SIZE, np.nan))
        bid_size = float(row.get(DataType.BID_SIZE, np.nan))
        ask = float(row.get(DataType.ASK, np.nan))
        bid = float(row.get(DataType.BID, np.nan))

        if np.isnan(ask_size) or np.isnan(bid_size) or np.isnan(ask) or np.isnan(bid):
            return np.nan

        vol = ask_size + bid_size
        if vol == 0:
            # Fall back to simple midpoint if sizes are zero
            return 0.5 * (ask + bid)

        weight_ask = ask_size / vol
        weight_bid = 1.0 - weight_ask
        return ask * weight_ask + bid * weight_bid
    except Exception:
        return np.nan


def compute_annualized_volatility(price_series: pd.Series, window_days: int = 21) -> pd.Series:
    """Compute rolling annualized volatility from a price series (daily or periodic).

    Args:
        price_series: pandas Series indexed by date with price values.
        window_days: rolling window in days for standard deviation.

    Returns:
        pandas Series of annualized volatility.
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    rolling_std = log_returns.rolling(window=window_days).std()
    annualized = rolling_std * np.sqrt(252)  # approximate trading days
    return annualized


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved pickle to %s", path)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---- ThetaData client wrapper ----
class ThetaDataFetcher:
    """A thin wrapper around ThetaClient focused on the specific queries we need."""

    def __init__(self, username: str, passwd: str, timeout: int = 15):
        self.username = username
        self.passwd = passwd
        self.timeout = timeout

    def _new_client(self) -> ThetaClient:
        return ThetaClient(username=self.username, passwd=self.passwd, timeout=self.timeout)

    def get_expirations(self, root: str) -> List[str]:
        """Return list of expirations (as returned by ThetaClient.get_expirations)."""
        client = self._new_client()
        with client.connect():
            expirations = client.get_expirations(root=root)
        logger.info("Fetched %d expirations for %s", len(expirations), root)
        return expirations

    def get_strikes(self, root: str, expirations: List[str]) -> Dict[str, pd.Series]:
        """Return a dictionary mapping expiration -> strikes (numeric series)."""
        result: Dict[str, pd.Series] = {}
        client = self._new_client()
        with client.connect():
            for exp in expirations:
                try:
                    data = client.get_strikes(root=root, exp=exp)
                    result[exp] = pd.to_numeric(data)
                except Exception as exc:
                    logger.warning("Failed to fetch strikes for %s exp %s: %s", root, exp, exc)
                    result[exp] = pd.Series(dtype=float)
        return result

    def get_hist_stock(self, root: str, trading_days: List[datetime], interval_size_ms: int) -> Dict[datetime, float]:
        """Fetch historical underlying quotes and return a mapping date -> price (weighted mid)."""
        client = self._new_client()
        underlying: Dict[datetime, float] = {}
        with client.connect():
            for tdate in trading_days:
                try:
                    # ThetaClient.get_hist_stock returns a DataFrame-like structure
                    data = client.get_hist_stock(
                        req=SecType.QUOTE,
                        root=root,
                        date_range=DateRange(tdate, tdate),
                        interval_size=interval_size_ms,
                    )

                    # Convert rows to weighted mid price; safe fallback to NaN
                    mid_prices = data.apply(weighted_mid_price_row, axis=1)
                    # Choose a representative value. Here we take the last valid mid-price of the day.
                    last_valid = mid_prices[~mid_prices.isna()].iloc[-1] if not mid_prices.dropna().empty else np.nan
                    underlying[tdate] = float(last_valid) if not np.isnan(last_valid) else np.nan
                except Exception as exc:
                    logger.warning("Failed to fetch hist stock for %s on %s: %s", root, tdate, exc)
                    underlying[tdate] = np.nan
        return underlying


# ---- Plotting ----
def plot_price_and_volatility(spot_df: pd.DataFrame, vol_col: str = "vol") -> None:
    """Plot price and volatility together. Expects spot_df to have 'price' and vol_col."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(spot_df.index, spot_df["price"], label="Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")

    ax2 = ax.twinx()
    ax2.plot(spot_df.index, spot_df[vol_col] * 100, label="Vol (%)", linestyle="--")
    ax2.set_ylabel("Volatility (%)")

    plt.title("Underlying Price and Annualized Volatility")
    fig.tight_layout()
    plt.show()


# ---- Main CLI / workflow ----
def main(
    root: str = "MSFT",
    start: str = "2012-06-01",
    end: str = "2022-11-14",
    interval_ms: int = 60 * 60000,
    save_dir: Optional[str] = ".",
    fetch_expirations_flag: bool = True,
    fetch_strikes_flag: bool = True,
    fetch_underlying_flag: bool = True,
):
    creds = get_env_credentials()
    fetcher = ThetaDataFetcher(username=creds["username"], passwd=creds["passwd"])

    # Parse date range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    trading_days = pd.date_range(start=start_dt, end=end_dt, freq="B").to_pydatetime().tolist()

    # ---- Expirations ----
    expirations = []
    if fetch_expirations_flag:
        try:
            expirations = fetcher.get_expirations(root)
            if save_dir:
                save_pickle(expirations, os.path.join(save_dir, f"{root}_expirations.pkl"))
        except Exception as exc:
            logger.error("Error fetching expirations: %s", exc)

    # ---- Strikes ----
    all_strikes = {}
    if fetch_strikes_flag and expirations:
        try:
            all_strikes = fetcher.get_strikes(root, expirations)
            if save_dir:
                save_pickle(all_strikes, os.path.join(save_dir, f"{root}_strikes.pkl"))
        except Exception as exc:
            logger.error("Error fetching strikes: %s", exc)

    # ---- Underlying historical ----
    underlying = {}
    if fetch_underlying_flag:
        try:
            underlying = fetcher.get_hist_stock(root, trading_days, interval_ms)
            if save_dir:
                save_pickle(underlying, os.path.join(save_dir, f"{root}_underlying.pkl"))
        except Exception as exc:
            logger.error("Error fetching underlying: %s", exc)

    # ---- Post-process and plot ----
    if underlying:
        spot = pd.DataFrame(list(underlying.items()), columns=["trade_date", "price"])
        spot.set_index("trade_date", inplace=True)
        spot.index = pd.to_datetime(spot.index)
        spot = spot.dropna()

        if spot.empty:
            logger.warning("No spot data available to compute volatility.")
            return

        spot["vol"] = compute_annualized_volatility(spot["price"], window_days=21)
        logger.info("Computed volatility; last rows:\n%s", spot.tail().to_string())

        plot_price_and_volatility(spot, vol_col="vol")
    else:
        logger.warning("No underlying data fetched; skipping plotting.")


# ---- CLI argument parsing (simple) ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and plot data from ThetaData")
    parser.add_argument("--root", default="MSFT", help="Underlying ticker/root")
    parser.add_argument("--start", default="2012-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-11-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval-ms", type=int, default=60 * 60000, help="Interval size in ms")
    parser.add_argument("--save-dir", default=".", help="Directory to save pickles (optional)")
    parser.add_argument("--no-expirations", action="store_true", help="Do not fetch expirations")
    parser.add_argument("--no-strikes", action="store_true", help="Do not fetch strikes")
    parser.add_argument("--no-underlying", action="store_true", help="Do not fetch underlying data")

    args = parser.parse_args()

    main(
        root=args.root,
        start=args.start,
        end=args.end,
        interval_ms=args.interval_ms,
        save_dir=args.save_dir,
        fetch_expirations_flag=not args.no_expirations,
        fetch_strikes_flag=not args.no_strikes,
        fetch_underlying_flag=not args.no_underlying,
    )
