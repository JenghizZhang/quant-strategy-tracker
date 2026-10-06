from pathlib import Path
import json

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# Strategy 1 — 双轨趋势
#
# Signal at today's close:
# 1. NDX Close > NDX MA30
#       -> QQQ
#
# 2. NDX Close <= NDX MA30
#    AND SPX MA50 > SPX MA200
#       -> SPY
#
# 3. Otherwise
#       -> AGG
#
# Execution:
#       Next trading day's OPEN
#
# Transaction cost:
#       0
#
# Backtest:
#       From first trading day of 2016
# ============================================================

DOWNLOAD_START = "2014-01-01"
BACKTEST_START = pd.Timestamp("2016-01-01")

NDX = "^NDX"
SPX = "^GSPC"

ASSETS = ["QQQ", "SPY", "AGG"]

OUTPUT_DIR = Path("output")


# ------------------------------------------------------------
# Download adjusted historical prices
# auto_adjust=True:
# prices are adjusted for splits/dividends
# ------------------------------------------------------------
def download_history(ticker):
    print(f"Downloading {ticker}...")

    df = yf.Ticker(ticker).history(
        start=DOWNLOAD_START,
        auto_adjust=True,
        actions=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    if "Open" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"Missing Open/Close data for {ticker}")

    df = df[["Open", "Close"]].copy()

    # Remove timezone while preserving the trading date.
    idx = pd.DatetimeIndex(df.index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    df.index = idx.normalize()

    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna()

    return df


# ------------------------------------------------------------
# Generate close-of-day signals
# ------------------------------------------------------------
def build_signals(ndx, spx):
    signals = pd.concat(
        [
            ndx["Close"].rename("NDX Close"),
            spx["Close"].rename("SPX Close"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    signals["NDX MA30"] = (
        signals["NDX Close"]
        .rolling(window=30, min_periods=30)
        .mean()
    )

    signals["SPX MA50"] = (
        signals["SPX Close"]
        .rolling(window=50, min_periods=50)
        .mean()
    )

    signals["SPX MA200"] = (
        signals["SPX Close"]
        .rolling(window=200, min_periods=200)
        .mean()
    )

    signals = signals.dropna().copy()

    condition_qqq = (
        signals["NDX Close"] > signals["NDX MA30"]
    )

    condition_spy = (
        (~condition_qqq)
        & (signals["SPX MA50"] > signals["SPX MA200"])
    )

    signals["Signal"] = np.select(
        [
            condition_qqq,
            condition_spy,
        ],
        [
            "QQQ",
            "SPY",
        ],
        default="AGG",
    )

    signals["Reason"] = np.select(
        [
            condition_qqq,
            condition_spy,
        ],
        [
            "NDX > MA30 → QQQ",
            "NDX <= MA30 and SPX MA50 > MA200 → SPY",
        ],
        default="NDX <= MA30 and SPX MA50 <= MA200 → AGG",
    )

    signals["NDX > MA30"] = condition_qqq

    signals["SPX MA50 > MA200"] = (
        signals["SPX MA50"] > signals["SPX MA200"]
    )

    return signals


# ------------------------------------------------------------
# Backtest
#
# Important:
#
# Signal is generated at today's CLOSE.
# The new position is entered NEXT trading day at OPEN.
#
# Therefore:
#
# Previous position receives:
#     previous close -> next open
#
# New position receives:
#     next open -> next close
#
# This avoids look-ahead bias.
# ------------------------------------------------------------
def run_backtest(signals, prices):
    # Use only dates available for:
    # NDX/SPX signal + all 3 ETFs
    common_index = signals.index.copy()

    for ticker in ASSETS:
        common_index = common_index.intersection(
            prices[ticker].index
        )

    common_index = common_index.sort_values()

    signals = signals.loc[common_index].copy()

    execution = pd.DataFrame(index=common_index)

    # Today's execution uses yesterday's closing signal.
    execution["Target"] = signals["Signal"].shift(1)

    execution["Reason"] = signals["Reason"].shift(1)

    execution["Signal Date"] = (
        pd.Series(
            common_index,
            index=common_index,
        )
        .shift(1)
    )

    execution = execution.loc[
        execution.index >= BACKTEST_START
    ].copy()

    execution = execution.dropna(
        subset=["Target", "Signal Date"]
    )

    if execution.empty:
        raise RuntimeError(
            "No backtest dates available after 2016."
        )

    equity = 1.0
    holding = None
    previous_date = None

    daily_rows = []
    position_history = []

    for date, row in execution.iterrows():
        target = row["Target"]
        signal_date = row["Signal Date"]
        reason = row["Reason"]

        # First trading day:
        # Start with $1 at the opening price.
        if previous_date is None:
            old_holding = "CASH"

            holding = target

            open_price = prices[holding].at[
                date,
                "Open",
            ]

            close_price = prices[holding].at[
                date,
                "Close",
            ]

            equity *= close_price / open_price

            position_history.append(
                {
                    "Change": f"{old_holding} → {holding}",
                    "Signal Date": signal_date,
                    "Execute Date": date,
                    "From": old_holding,
                    "To": holding,
                    "Reason": reason,
                }
            )

        else:
            # ------------------------------------------------
            # Overnight return belongs to the OLD position.
            # ------------------------------------------------
            old_open = prices[holding].at[
                date,
                "Open",
            ]

            old_previous_close = prices[holding].at[
                previous_date,
                "Close",
            ]

            equity *= (
                old_open / old_previous_close
            )

            # ------------------------------------------------
            # Switch at today's OPEN if signal changed.
            # ------------------------------------------------
            if target != holding:
                old_holding = holding
                holding = target

                position_history.append(
                    {
                        "Change": (
                            f"{old_holding} → {holding}"
                        ),
                        "Signal Date": signal_date,
                        "Execute Date": date,
                        "From": old_holding,
                        "To": holding,
                        "Reason": reason,
                    }
                )

            # ------------------------------------------------
            # Intraday return belongs to the NEW/current
            # position.
            # ------------------------------------------------
            current_open = prices[holding].at[
                date,
                "Open",
            ]

            current_close = prices[holding].at[
                date,
                "Close",
            ]

            equity *= (
                current_close / current_open
            )

        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Holding": holding,
                "Target": target,
                "Signal Date": signal_date,
                "Signal Reason": reason,
            }
        )

        previous_date = date

    daily = pd.DataFrame(
        daily_rows
    ).set_index("Date")

    history = pd.DataFrame(position_history)

    return daily, history


# ------------------------------------------------------------
# Buy & Hold benchmark
#
# Benchmark also begins at the OPEN of the same first
# backtest trading day.
# ------------------------------------------------------------
def buy_and_hold_equity(price_df, dates):
    first_date = dates[0]

    initial_open = price_df.at[
        first_date,
        "Open",
    ]

    equity = (
        price_df.loc[dates, "Close"]
        / initial_open
    )

    return equity


# ------------------------------------------------------------
# Monthly / annual returns
# ------------------------------------------------------------
def calculate_period_returns(equity, period):
    if period == "month":
        end_values = equity.groupby(
            equity.index.to_period("M")
        ).last()

    elif period == "year":
        end_values = equity.groupby(
            equity.index.year
        ).last()

    else:
        raise ValueError(
            f"Unknown period: {period}"
        )

    returns = end_values.pct_change()

    # First period begins from initial equity = 1
    returns.iloc[0] = (
        end_values.iloc[0] - 1.0
    )

    return returns


# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------
def calculate_metrics(equity):
    latest_year = equity.index[-1].year

    prior_year_values = equity[
        equity.index.year < latest_year
    ]

    if len(prior_year_values) > 0:
        ytd_base = prior_year_values.iloc[-1]
    else:
        ytd_base = 1.0

    ytd_return = (
        equity.iloc[-1] / ytd_base
    ) - 1.0

    since_2016 = equity.iloc[-1] - 1.0

    years = (
        equity.index[-1]
        - equity.index[0]
    ).days / 365.25

    if years > 0:
        cagr = (
            equity.iloc[-1] ** (1 / years)
        ) - 1.0
    else:
        cagr = 0.0

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
    ) - 1.0

    max_drawdown = drawdown.min()

    return {
        "YTD Return": float(ytd_return),
        "Since 2016": float(since_2016),
        "CAGR": float(cagr),
        "Max Drawdown": float(max_drawdown),
    }


def percent(value):
    return f"{value * 100:.2f}%"


def main():
    print("=" * 70)
    print("Strategy 1 — 双轨趋势")
    print("=" * 70)

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------
    ndx = download_history(NDX)
    spx = download_history(SPX)

    prices = {
        ticker: download_history(ticker)
        for ticker in ASSETS
    }

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------
    signals = build_signals(
        ndx,
        spx,
    )

    # --------------------------------------------------------
    # Strategy backtest
    # --------------------------------------------------------
    strategy_daily, position_history = (
        run_backtest(
            signals,
            prices,
        )
    )

    dates = strategy_daily.index

    strategy_equity = (
        strategy_daily["Equity"]
    )

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------
    qqq_equity = buy_and_hold_equity(
        prices["QQQ"],
        dates,
    )

    spy_equity = buy_and_hold_equity(
        prices["SPY"],
        dates,
    )

    # --------------------------------------------------------
    # Monthly performance
    # --------------------------------------------------------
    monthly = pd.DataFrame(
        {
            "Strategy 1": (
                calculate_period_returns(
                    strategy_equity,
                    "month",
                )
            ),
            "QQQ": (
                calculate_period_returns(
                    qqq_equity,
                    "month",
                )
            ),
            "SPY": (
                calculate_period_returns(
                    spy_equity,
                    "month",
                )
            ),
        }
    )

    monthly.index = (
        monthly.index.astype(str)
    )

    monthly.index.name = "Month"

    # --------------------------------------------------------
    # Annual performance
    # --------------------------------------------------------
    annual = pd.DataFrame(
        {
            "Strategy 1": (
                calculate_period_returns(
                    strategy_equity,
                    "year",
                )
            ),
            "QQQ": (
                calculate_period_returns(
                    qqq_equity,
                    "year",
                )
            ),
            "SPY": (
                calculate_period_returns(
                    spy_equity,
                    "year",
                )
            ),
        }
    )

    annual.index.name = "Year"

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------
    summary = {
        "Strategy 1": calculate_metrics(
            strategy_equity
        ),
        "QQQ": calculate_metrics(
            qqq_equity
        ),
        "SPY": calculate_metrics(
            spy_equity
        ),
    }

    # --------------------------------------------------------
    # Current status
    # --------------------------------------------------------
    latest_date = dates[-1]

    current_holding = (
        strategy_daily.iloc[-1]["Holding"]
    )

    today_signal = signals.at[
        latest_date,
        "Signal",
    ]

    next_holding = today_signal

    current_status = {
        "Signal Date": str(
            latest_date.date()
        ),
        "Current Holding": (
            current_holding
        ),
        "Today's Signal": (
            today_signal
        ),
        "Next Holding": (
            next_holding
        ),
        "Pending Change": (
            current_holding
            != next_holding
        ),
        "NDX Close": float(
            signals.at[
                latest_date,
                "NDX Close",
            ]
        ),
        "NDX MA30": float(
            signals.at[
                latest_date,
                "NDX MA30",
            ]
        ),
        "NDX > MA30": bool(
            signals.at[
                latest_date,
                "NDX > MA30",
            ]
        ),
        "SPX MA50": float(
            signals.at[
                latest_date,
                "SPX MA50",
            ]
        ),
        "SPX MA200": float(
            signals.at[
                latest_date,
                "SPX MA200",
            ]
        ),
        "SPX MA50 > MA200": bool(
            signals.at[
                latest_date,
                "SPX MA50 > MA200",
            ]
        ),
        "Reason": signals.at[
            latest_date,
            "Reason",
        ],
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_daily.to_csv(
        OUTPUT_DIR
        / "strategy1_daily.csv"
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "strategy1_monthly.csv"
    )

    annual.to_csv(
        OUTPUT_DIR
        / "strategy1_annual.csv"
    )

    position_history.to_csv(
        OUTPUT_DIR
        / "strategy1_position_history.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "strategy1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "current_status": (
                    current_status
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print()
    print("BACKTEST PERIOD")
    print(
        f"{dates[0].date()}"
        f" → "
        f"{dates[-1].date()}"
    )

    print()
    print("=" * 70)
    print("CURRENT STATUS")
    print("=" * 70)

    print(
        "Current Holding :",
        current_holding,
    )

    print(
        "Today's Signal  :",
        today_signal,
    )

    print(
        "Next Holding    :",
        next_holding,
    )

    print(
        "Pending Change  :",
        (
            "YES"
            if current_holding
            != next_holding
            else "NO"
        ),
    )

    print()
    print(
        f"NDX Close       : "
        f"{current_status['NDX Close']:.2f}"
    )

    print(
        f"NDX MA30        : "
        f"{current_status['NDX MA30']:.2f}"
    )

    print(
        "NDX > MA30      :",
        current_status[
            "NDX > MA30"
        ],
    )

    print()
    print(
        f"SPX MA50        : "
        f"{current_status['SPX MA50']:.2f}"
    )

    print(
        f"SPX MA200       : "
        f"{current_status['SPX MA200']:.2f}"
    )

    print(
        "MA50 > MA200    :",
        current_status[
            "SPX MA50 > MA200"
        ],
    )

    print()
    print(
        "Reason           :",
        current_status["Reason"],
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, metrics in summary.items():
        print()
        print(name)

        print(
            "  YTD Return    :",
            percent(
                metrics["YTD Return"]
            ),
        )

        print(
            "  Since 2016    :",
            percent(
                metrics["Since 2016"]
            ),
        )

        print(
            "  CAGR          :",
            percent(
                metrics["CAGR"]
            ),
        )

        print(
            "  Max Drawdown  :",
            percent(
                metrics["Max Drawdown"]
            ),
        )

    print()
    print("=" * 70)
    print("LATEST 12 MONTHS")
    print("=" * 70)

    print(
        monthly.tail(12)
        .to_string(
            float_format=lambda x: (
                f"{x * 100:.2f}%"
            )
        )
    )

    print()
    print("=" * 70)
    print("ANNUAL RETURNS")
    print("=" * 70)

    print(
        annual.to_string(
            float_format=lambda x: (
                f"{x * 100:.2f}%"
            )
        )
    )

    print()
    print(
        f"Position changes: "
        f"{len(position_history)}"
    )

    print()
    print(
        "✅ Strategy 1 backtest completed"
    )


if __name__ == "__main__":
    main()
