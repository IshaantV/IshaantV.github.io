#!/usr/bin/env python3
# ===============================================================
# Temperature Data Processing, Modeling and Decomposition
# ===============================================================
# - Loads max/min daily temperature CSVs from URLs
# - Cleans & merges datasets
# - Computes average daily temperature
# - Converts dates to ordinals for modeling and restores them
# - Fits a seasonal + linear model using scipy.optimize.curve_fit
# - Performs seasonal-trend decomposition using statsmodels
# - Produces plots for observed vs fitted and decomposition components
#
# Author: (Your Name)
# Date: (Auto-generated)
# ===============================================================

# ---- Standard imports ----
import os
import sys
import math
import logging
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# ---- Scientific imports ----
from scipy import interpolate
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose

# ---- Constants & URLs ----
MAX_TEMP_URL = "https://raw.githubusercontent.com/ASXPortfolio/jupyter-notebooks-data/main/maximum_temperature.csv"
MIN_TEMP_URL = "https://raw.githubusercontent.com/ASXPortfolio/jupyter-notebooks-data/main/minimum_temperature.csv"

# ---- Logging configuration ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("temp_model")


# ---------------------------------------------------------------
# Data loading & cleaning utilities
# ---------------------------------------------------------------
def load_temperature_data(url: str, temp_type: str) -> pd.DataFrame:
    """
    Load CSV from url and prepare a cleaned DataFrame with Date index
    and a single column 'Tmax' or 'Tmin'.

    Args:
        url: URL to CSV
        temp_type: 'max' or 'min'

    Returns:
        DataFrame indexed by datetime with column 'Tmax' or 'Tmin'.
    """
    logger.info("Loading %s data from: %s", temp_type.upper(), url)
    df = pd.read_csv(url)

    # Create Date column using Year/Month/Day
    def _row_to_datetime(row):
        return dt.datetime(int(row["Year"]), int(row["Month"]), int(row["Day"]))

    df["Date"] = df.apply(_row_to_datetime, axis=1)
    df.set_index("Date", inplace=True)

    # Drop columns we don't need (keeps the temperature column)
    drop_cols = [0, 1, 2, 3, 4, 6, 7]  # same as original indices
    df.drop(df.columns[drop_cols], axis=1, inplace=True)

    # Rename the remaining column for clarity
    if temp_type == "max":
        df.rename(columns={"Maximum temperature (Degree C)": "Tmax"}, inplace=True)
    elif temp_type == "min":
        df.rename(columns={"Minimum temperature (Degree C)": "Tmin"}, inplace=True)
    else:
        raise ValueError("temp_type must be 'max' or 'min'")

    logger.info("Loaded %d rows of %s data.", len(df), temp_type.upper())
    return df


def merge_and_compute_average(max_df: pd.DataFrame, min_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Tmax and Tmin DataFrames and compute average daily temperature T.

    Returns DataFrame with columns ['Tmax', 'Tmin', 'T'] indexed by Date.
    """
    logger.info("Merging Tmax and Tmin DataFrames...")
    merged = max_df.merge(min_df, how="inner", left_index=True, right_index=True)
    logger.info("Merged size: %d rows", len(merged))

    # Compute average temperature column 'T'
    merged["T"] = merged.apply(lambda r: (r["Tmax"] + r["Tmin"]) / 2, axis=1)
    logger.info("Computed average temperature 'T' for merged data.")
    # Drop NA rows to keep modeling clean
    before = len(merged)
    merged = merged.dropna()
    after = len(merged)
    logger.info("Dropped %d rows with missing values (if any).", before - after)
    return merged


# ---------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------
def T_model(x: np.ndarray, a: float, b: float, alpha: float, theta: float) -> np.ndarray:
    """
    Seasonal temperature model: a + b*x + alpha * sin(omega*x + theta)
    where omega = 2*pi / 365.25
    x should be in days (e.g. ordinal days relative to a reference).
    """
    omega = 2.0 * np.pi / 365.25
    return a + b * x + alpha * np.sin(omega * x + theta)


def dT_model(x: np.ndarray, a: float, b: float, alpha: float, theta: float) -> np.ndarray:
    """
    Derivative of T_model (for reference/analysis).
    """
    omega = 2.0 * np.pi / 365.25
    return b + alpha * omega * np.cos(omega * x + theta)


# ---------------------------------------------------------------
# Modeling workflow helpers
# ---------------------------------------------------------------
def convert_index_to_ordinal(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Convert a DatetimeIndex to ordinal integers (days), returning the modified
    DataFrame and the first ordinal (used as reference).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")
    ordinals = df.index.map(dt.datetime.toordinal)
    first_ord = int(ordinals[0])
    df_ord = df.copy(deep=True)
    df_ord.index = ordinals
    logger.info("Converted DatetimeIndex to ordinal days. first_ord=%d", first_ord)
    return df_ord, first_ord


def restore_datetime_index_from_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert an ordinal-day index back to a DatetimeIndex. Modifies a copy.
    """
    df_copy = df.copy(deep=True)
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = df_copy.index.map(dt.datetime.fromordinal)
        logger.info("Restored DatetimeIndex from ordinals.")
    return df_copy


def fit_seasonal_model(df: pd.DataFrame, first_ord: int, guess: list | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fit the T_model to observed data using curve_fit.

    Returns: (df_with_model, params, cov_matrix)
    - df_with_model has column 'model_fit' appended
    - params is array [a, b, alpha, theta]
    """
    logger.info("Fitting seasonal + linear model using scipy.optimize.curve_fit...")
    x = (df.index.astype(int) - first_ord).astype(float)
    y = df["T"].values.astype(float)

    if guess is None:
        guess = [np.nanmean(y), 0.0, 5.0, 0.0]  # baseline guess

    try:
        params, cov = curve_fit(T_model, x, y, p0=guess, maxfev=20000)
        logger.info("curve_fit succeeded. Parameters: %s", np.array2string(params, precision=6))
    except Exception as exc:
        logger.warning("curve_fit failed: %s. Falling back to initial guess and not attaching model.", exc)
        params = np.array(guess, dtype=float)
        cov = np.zeros((len(guess), len(guess)))
        # still attach a model based on the guess
    df_with = df.copy(deep=True)
    df_with["model_fit"] = T_model(x, *params)
    return df_with, params, cov


# ---------------------------------------------------------------
# Decomposition & plotting utilities
# ---------------------------------------------------------------
def perform_seasonal_decompose(df: pd.DataFrame, column: str = "T", period: int = 12) -> object:
    """
    Perform seasonal-trend decomposition on a resampled (monthly) series.
    Note: seasonal_decompose expects evenly spaced data and a specified period.
    We resample to monthly averages here before decomposition.
    """
    logger.info("Preparing data for seasonal decomposition (monthly resample)...")
    monthly = df[column].resample("M").mean()
    if monthly.isna().all():
        raise ValueError("Resampled monthly series is empty or all NaNs.")
    # period argument: for monthly series, annual cycle is 12
    logger.info("Running seasonal_decompose with period=%d ...", period)
    decomposition = seasonal_decompose(monthly, model="additive", period=period, extrapolate_trend="freq")
    logger.info("Decomposition complete.")
    return decomposition


def plot_observed_vs_model(df: pd.DataFrame, recent_n: int = 2000):
    """Plot observed average temperature vs the fitted model on the most recent N points."""
    logger.info("Plotting observed vs fitted model for the most recent %d points...", recent_n)
    df_plot = df.tail(recent_n)
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot.index, df_plot["T"], label="Observed T", marker="o", markersize=3, linewidth=1)
    if "model_fit" in df_plot.columns:
        plt.plot(df_plot.index, df_plot["model_fit"], label="Model Fit", linestyle="--", linewidth=2)
    plt.title("Observed vs Fitted Temperature (Recent Period)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decomposition_result(decomp):
    """Plot the trend/seasonal/residual components returned by seasonal_decompose."""
    logger.info("Plotting decomposition result (trend, seasonal, resid)...")
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle("Seasonal-Trend Decomposition", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# Optional: simple anomaly detection vs model
# ---------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame, threshold_std: float = 2.0) -> pd.DataFrame:
    """
    Flag days where the residual (T - model_fit) exceeds threshold_std * residual_std.
    Returns a copy of df with an 'anomaly' boolean column and 'residual' column.
    """
    logger.info("Detecting anomalies using threshold %.2f sigma...", threshold_std)
    df_copy = df.copy(deep=True)
    if "model_fit" not in df_copy.columns:
        raise ValueError("DataFrame must contain 'model_fit' to detect anomalies.")
    df_copy["residual"] = df_copy["T"] - df_copy["model_fit"]
    resid_std = df_copy["residual"].std()
    thresh = threshold_std * resid_std
    df_copy["anomaly"] = df_copy["residual"].abs() > thresh
    logger.info("Anomaly detection complete. Found %d anomalies.", int(df_copy["anomaly"].sum()))
    return df_copy


# ---------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------
def main():
    logger.info("=== Temperature analysis & modeling started ===")

    # 1) Load data
    max_df = load_temperature_data(MAX_TEMP_URL, "max")
    min_df = load_temperature_data(MIN_TEMP_URL, "min")

    # 2) Merge & compute average temperature
    temps = merge_and_compute_average(max_df, min_df)

    # 3) Extract average temperature series DataFrame (T only)
    temp_t = temps[["T"]].copy(deep=True)
    logger.info("Prepared temperature series with %d rows.", len(temp_t))

    # 4) Convert to ordinal days for model fitting
    try:
        temp_t_ord, first_ord = convert_index_to_ordinal(temp_t)
    except Exception as e:
        logger.error("Index conversion failed: %s", e)
        return

    # 5) Fit seasonal model automatically
    initial_guess = [temp_t_ord["T"].mean(), 0.0, 5.0, 0.0]
    temp_t_with_model, params, cov = fit_seasonal_model(temp_t_ord, first_ord, guess=initial_guess)
    logger.info("Fitted params: a=%.4f, b=%.6e, alpha=%.4f, theta=%.4f", *params)

    # 6) Restore datetime index
    temp_t_restored = restore_datetime_index_from_ordinal(temp_t_with_model)

    # 7) Plot observed vs fitted model
    plot_observed_vs_model(temp_t_restored, recent_n=2000)

    # 8) Decomposition (monthly resample)
    decomposition = None
    try:
        decomposition = perform_seasonal_decompose(temp_t_restored, column="T", period=12)
        plot_decomposition_result(decomposition)
    except Exception as exc:
        logger.warning("Seasonal decomposition failed or was skipped: %s", exc)

    # 9) Anomaly detection (optional)
    try:
        anomalies_df = detect_anomalies(temp_t_restored, threshold_std=2.5)
        # Print a small sample of anomalies, if any
        n_anom = int(anomalies_df["anomaly"].sum())
        if n_anom > 0:
            logger.info("Sample anomalies (up to 10):")
            print(anomalies_df[anomalies_df["anomaly"]].head(10).to_string())
        else:
            logger.info("No anomalies detected at the chosen threshold.")
    except Exception as exc:
        logger.warning("Anomaly detection skipped or failed: %s", exc)

    logger.info("=== Temperature analysis & modeling finished ===")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()
