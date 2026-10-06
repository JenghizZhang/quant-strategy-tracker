from pathlib import Path
import json

import numpy as np
import pandas as pd

from market_data import (
    get_latest_completed_session,
    load_daily_history,
)


# ============================================================
# Strategy 1 — 双轨趋势
#
# Signal at today's CLOSE:
#
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
#
# Latest market data priority:
#       Daily
#       -> 5m regular-session fallback
#       -> 1m regular-session fallback
#       -> Fail safely
# ============================================================


DOWNLOAD_START = "2014-01-01"
BACKTEST_START = pd.Timestamp("2016-01-01")

NDX = "^NDX"
SPX = "^GSPC"

ASSETS = [
    "QQQ",
    "SPY",
    "AGG",
]

OUTPUT_DIR = Path("output")


# ============================================================
# SIGNALS
# ============================================================

def build_signals(ndx, spx):
    """
    Build Strategy 1 signals from daily closing data.
    """

    signals = pd.concat(
        [
            ndx["Close"].rename("NDX Close"),
            spx["Close"].rename("SPX Close"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    # ----------------------------------------
    # Nasdaq 100 MA30
    # ----------------------------------------
    signals["NDX MA30"] = (
        signals["NDX Close"]
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    # ----------------------------------------
    # S&P 500 MA50
    # ----------------------------------------
    signals["SPX MA50"] = (
        signals["SPX Close"]
        .rolling(
            window=50,
            min_periods=50,
        )
        .mean()
    )

    # ----------------------------------------
    # S&P 500 MA200
    # ----------------------------------------
    signals["SPX MA200"] = (
        signals["SPX Close"]
        .rolling(
            window=200,
            min_periods=200,
        )
        .mean()
    )

    signals = signals.dropna().copy()

    # ----------------------------------------
    # Rule 1:
    # NDX > MA30
    # -> QQQ
    # ----------------------------------------
    condition_qqq = (
        signals["NDX Close"]
        > signals["NDX MA30"]
    )

    # ----------------------------------------
    # Rule 2:
    # NDX <= MA30
    # AND
    # SPX MA50 > MA200
    # -> SPY
    # ----------------------------------------
    condition_spy = (
        (~condition_qqq)
        & (
            signals["SPX MA50"]
            > signals["SPX MA200"]
        )
    )

    # ----------------------------------------
    # Otherwise:
    # -> AGG
    # ----------------------------------------
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
            (
                "NDX <= MA30 and "
                "SPX MA50 > MA200 → SPY"
            ),
        ],
        default=(
            "NDX <= MA30 and "
            "SPX MA50 <= MA200 → AGG"
        ),
    )

    signals["NDX > MA30"] = (
        condition_qqq
    )

    signals["SPX MA50 > MA200"] = (
        signals["SPX MA50"]
        > signals["SPX MA200"]
    )

    return signals


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    signals,
    prices,
):
    """
    Strategy execution rule:

        Signal is generated at today's CLOSE.

        New position is entered at
        NEXT trading day's OPEN.

    Therefore:

        Previous position receives:
            previous close -> next open

        New/current position receives:
            next open -> next close

    This avoids look-ahead bias.
    """

    # ----------------------------------------
    # Only use dates available across:
    #
    # NDX
    # SPX
    # QQQ
    # SPY
    # AGG
    # ----------------------------------------
    common_index = (
        signals.index.copy()
    )

    for ticker in ASSETS:
        common_index = (
            common_index.intersection(
                prices[ticker].index
            )
        )

    common_index = (
        common_index.sort_values()
    )

    signals = signals.loc[
        common_index
    ].copy()

    execution = pd.DataFrame(
        index=common_index
    )

    # ----------------------------------------
    # Today's open executes yesterday's
    # closing signal.
    # ----------------------------------------
    execution["Target"] = (
        signals["Signal"].shift(1)
    )

    execution["Reason"] = (
        signals["Reason"].shift(1)
    )

    execution["Signal Date"] = (
        pd.Series(
            common_index,
            index=common_index,
        )
        .shift(1)
    )

    # ----------------------------------------
    # Start from 2016
    # ----------------------------------------
    execution = execution.loc[
        execution.index
        >= BACKTEST_START
    ].copy()

    execution = (
        execution.dropna(
            subset=[
                "Target",
                "Signal Date",
            ]
        )
    )

    if execution.empty:
        raise RuntimeError(
            "No backtest dates "
            "available after 2016."
        )

    equity = 1.0

    holding = None
    previous_date = None

    daily_rows = []
    position_history = []

    for date, row in execution.iterrows():

        target = row["Target"]
        signal_date = row[
            "Signal Date"
        ]
        reason = row["Reason"]

        # ====================================
        # First trading day
        # ====================================

        if previous_date is None:

            old_holding = "CASH"

            holding = target

            open_price = (
                prices[holding]
                .at[
                    date,
                    "Open",
                ]
            )

            close_price = (
                prices[holding]
                .at[
                    date,
                    "Close",
                ]
            )

            # Start with $1 at the open.
            equity *= (
                close_price
                / open_price
            )

            position_history.append(
                {
                    "Change": (
                        f"{old_holding}"
                        f" → "
                        f"{holding}"
                    ),
                    "Signal Date": (
                        signal_date
                    ),
                    "Execute Date": date,
                    "From": old_holding,
                    "To": holding,
                    "Reason": reason,
                }
            )

        # ====================================
        # All later trading days
        # ====================================

        else:

            # --------------------------------
            # Overnight:
            #
            # yesterday close
            # ->
            # today open
            #
            # belongs to OLD position.
            # --------------------------------

            old_open = (
                prices[holding]
                .at[
                    date,
                    "Open",
                ]
            )

            old_previous_close = (
                prices[holding]
                .at[
                    previous_date,
                    "Close",
                ]
            )

            equity *= (
                old_open
                / old_previous_close
            )

            # --------------------------------
            # Switch exactly at today's OPEN.
            # --------------------------------

            if target != holding:

                old_holding = holding
                holding = target

                position_history.append(
                    {
                        "Change": (
                            f"{old_holding}"
                            f" → "
                            f"{holding}"
                        ),
                        "Signal Date": (
                            signal_date
                        ),
                        "Execute Date": (
                            date
                        ),
                        "From": (
                            old_holding
                        ),
                        "To": holding,
                        "Reason": reason,
                    }
                )

            # --------------------------------
            # Today's open -> today's close
            # belongs to NEW/current position.
            # --------------------------------

            current_open = (
                prices[holding]
                .at[
                    date,
                    "Open",
                ]
            )

            current_close = (
                prices[holding]
                .at[
                    date,
                    "Close",
                ]
            )

            equity *= (
                current_close
                / current_open
            )

        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Holding": holding,
                "Target": target,
                "Signal Date": (
                    signal_date
                ),
                "Signal Reason": (
                    reason
                ),
            }
        )

        previous_date = date

    daily = pd.DataFrame(
        daily_rows
    ).set_index("Date")

    history = pd.DataFrame(
        position_history
    )

    return (
        daily,
        history,
    )


# ============================================================
# BENCHMARKS
# ============================================================

def buy_and_hold_equity(
    price_df,
    dates,
):
    """
    Benchmark begins at the OPEN of
    the same first backtest trading day.
    """

    first_date = dates[0]

    initial_open = (
        price_df.at[
            first_date,
            "Open",
        ]
    )

    equity = (
        price_df
        .loc[
            dates,
            "Close",
        ]
        / initial_open
    )

    return equity


# ============================================================
# MONTHLY / ANNUAL PERFORMANCE
# ============================================================

def calculate_period_returns(
    equity,
    period,
):

    if period == "month":

        end_values = (
            equity.groupby(
                equity.index.to_period(
                    "M"
                )
            )
            .last()
        )

    elif period == "year":

        end_values = (
            equity.groupby(
                equity.index.year
            )
            .last()
        )

    else:
        raise ValueError(
            f"Unknown period: "
            f"{period}"
        )

    returns = (
        end_values.pct_change()
    )

    # ----------------------------------------
    # First period starts from initial
    # equity = 1.
    # ----------------------------------------
    returns.iloc[0] = (
        end_values.iloc[0]
        - 1.0
    )

    return returns


# ============================================================
# SUMMARY METRICS
# ============================================================

def calculate_metrics(
    equity,
):

    latest_year = (
        equity.index[-1].year
    )

    # ----------------------------------------
    # YTD base:
    # last trading day of prior year
    # ----------------------------------------
    prior_year_values = equity[
        equity.index.year
        < latest_year
    ]

    if len(prior_year_values) > 0:
        ytd_base = (
            prior_year_values.iloc[-1]
        )
    else:
        ytd_base = 1.0

    ytd_return = (
        equity.iloc[-1]
        / ytd_base
    ) - 1.0

    # ----------------------------------------
    # Total return since start
    # ----------------------------------------
    since_2016 = (
        equity.iloc[-1]
        - 1.0
    )

    # ----------------------------------------
    # CAGR
    # ----------------------------------------
    years = (
        (
            equity.index[-1]
            - equity.index[0]
        ).days
        / 365.25
    )

    if years > 0:

        cagr = (
            equity.iloc[-1]
            ** (1 / years)
        ) - 1.0

    else:

        cagr = 0.0

    # ----------------------------------------
    # Max Drawdown
    # ----------------------------------------
    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
    ) - 1.0

    max_drawdown = (
        drawdown.min()
    )

    return {
        "YTD Return": float(
            ytd_return
        ),
        "Since 2016": float(
            since_2016
        ),
        "CAGR": float(
            cagr
        ),
        "Max Drawdown": float(
            max_drawdown
        ),
    }


def percent(value):
    return (
        f"{value * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Strategy 1 — 双轨趋势"
    )
    print("=" * 70)

    # ========================================================
    # Identify latest COMPLETED market session
    # ========================================================

    session = (
        get_latest_completed_session()
    )

    target_date = (
        session["date"]
    )

    print()

    print(
        "Latest completed "
        "market session:",
        target_date.date(),
    )

    print()

    # ========================================================
    # DOWNLOAD MARKET DATA
    # ========================================================

    ndx, ndx_source, _ = (
        load_daily_history(
            NDX,
            DOWNLOAD_START,
            session=session,
        )
    )

    spx, spx_source, _ = (
        load_daily_history(
            SPX,
            DOWNLOAD_START,
            session=session,
        )
    )

    prices = {}
    price_sources = {}

    for ticker in ASSETS:

        df, source, _ = (
            load_daily_history(
                ticker,
                DOWNLOAD_START,
                session=session,
            )
        )

        prices[ticker] = df

        price_sources[
            ticker
        ] = source

    # ========================================================
    # BUILD SIGNALS
    # ========================================================

    signals = build_signals(
        ndx,
        spx,
    )

    # ========================================================
    # RUN STRATEGY
    # ========================================================

    (
        strategy_daily,
        position_history,
    ) = run_backtest(
        signals,
        prices,
    )

    dates = (
        strategy_daily.index
    )

    # ========================================================
    # CRITICAL FRESHNESS CHECK
    #
    # Never silently use yesterday's data
    # as today's current status.
    # ========================================================

    if dates[-1] != target_date:

        raise RuntimeError(
            "Backtest did not reach "
            "the latest completed "
            "market session. "
            f"Expected "
            f"{target_date.date()}, "
            f"got "
            f"{dates[-1].date()}."
        )

    strategy_equity = (
        strategy_daily["Equity"]
    )

    # ========================================================
    # BENCHMARKS
    # ========================================================

    qqq_equity = (
        buy_and_hold_equity(
            prices["QQQ"],
            dates,
        )
    )

    spy_equity = (
        buy_and_hold_equity(
            prices["SPY"],
            dates,
        )
    )

    # ========================================================
    # MONTHLY PERFORMANCE
    # ========================================================

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

    # ========================================================
    # ANNUAL PERFORMANCE
    # ========================================================

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

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary = {
        "Strategy 1": (
            calculate_metrics(
                strategy_equity
            )
        ),
        "QQQ": (
            calculate_metrics(
                qqq_equity
            )
        ),
        "SPY": (
            calculate_metrics(
                spy_equity
            )
        ),
    }

    # ========================================================
    # CURRENT STATUS
    # ========================================================

    latest_date = (
        dates[-1]
    )

    # ----------------------------------------
    # Position actually held during today's
    # trading session.
    # ----------------------------------------
    current_holding = (
        strategy_daily
        .iloc[-1]["Holding"]
    )

    # ----------------------------------------
    # Today's closing signal.
    #
    # This is what should be held starting
    # NEXT trading day's open.
    # ----------------------------------------
    today_signal = (
        signals.at[
            latest_date,
            "Signal",
        ]
    )

    next_holding = (
        today_signal
    )

    pending_change = (
        current_holding
        != next_holding
    )

    current_status = {

        "Data Date": str(
            latest_date.date()
        ),

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
            pending_change
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

        "Reason": (
            signals.at[
                latest_date,
                "Reason",
            ]
        ),

        "Data Sources": {

            "NDX": (
                ndx_source
            ),

            "SPX": (
                spx_source
            ),

            **price_sources,
        },
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

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
        / (
            "strategy1_"
            "position_history.csv"
        ),
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

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()

    print(
        "BACKTEST PERIOD"
    )

    print(
        f"{dates[0].date()}"
        f" → "
        f"{dates[-1].date()}"
    )

    # ========================================================
    # DATA SOURCES
    # ========================================================

    print()

    print("=" * 70)
    print("DATA SOURCES")
    print("=" * 70)

    print(
        f"NDX : "
        f"{ndx_source}"
    )

    print(
        f"SPX : "
        f"{spx_source}"
    )

    for ticker in ASSETS:

        print(
            f"{ticker:<3} : "
            f"{price_sources[ticker]}"
        )

    # ========================================================
    # CURRENT STATUS
    # ========================================================

    print()

    print("=" * 70)
    print("CURRENT STATUS")
    print("=" * 70)

    print(
        "Data Date       :",
        latest_date.date(),
    )

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
            if pending_change
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

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, metrics in (
        summary.items()
    ):

        print()
        print(name)

        print(
            "  YTD Return    :",
            percent(
                metrics[
                    "YTD Return"
                ]
            ),
        )

        print(
            "  Since 2016    :",
            percent(
                metrics[
                    "Since 2016"
                ]
            ),
        )

        print(
            "  CAGR          :",
            percent(
                metrics[
                    "CAGR"
                ]
            ),
        )

        print(
            "  Max Drawdown  :",
            percent(
                metrics[
                    "Max Drawdown"
                ]
            ),
        )

    # ========================================================
    # LATEST 12 MONTHS
    # ========================================================

    print()

    print("=" * 70)
    print("LATEST 12 MONTHS")
    print("=" * 70)

    print(
        monthly
        .tail(12)
        .to_string(
            float_format=(
                lambda x: (
                    f"{x * 100:.2f}%"
                )
            )
        )
    )

    # ========================================================
    # ANNUAL RETURNS
    # ========================================================

    print()

    print("=" * 70)
    print("ANNUAL RETURNS")
    print("=" * 70)

    print(
        annual.to_string(
            float_format=(
                lambda x: (
                    f"{x * 100:.2f}%"
                )
            )
        )
    )

    # ========================================================
    # POSITION HISTORY
    # ========================================================

    print()

    print(
        f"Position changes: "
        f"{len(position_history)}"
    )

    print()

    print(
        "✅ Strategy 1 "
        "backtest completed"
    )


if __name__ == "__main__":
    main()
