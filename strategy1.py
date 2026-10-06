from pathlib import Path
import json

import pandas as pd

from market_data import (
    get_latest_completed_session,
    load_daily_history,
)


# ============================================================
# Strategy 1 — 双轨趋势（交叉事件驱动）
#
# State:
#   1 = QQQ
#   2 = SPY
#   3 = AGG
#
# Entry / exit rules (signal at CLOSE):
#
# QQQ entry:
#   NDX Close crosses ABOVE NDX MA30
#
# QQQ exit:
#   NDX Close crosses BELOW / to-or-below NDX MA30
#
# SPY entry:
#   SPX MA50 crosses ABOVE SPX MA200
#   AND Rule 1 is not satisfied on that day
#   (NDX Close <= NDX MA30)
#
# SPY exit:
#   SPX MA50 crosses BELOW / to-or-below SPX MA200
#
# State transition:
#   AGG -> QQQ : QQQ entry event
#   AGG -> SPY : SPY entry event and Rule 1 not satisfied
#   QQQ -> AGG : QQQ exit event
#   SPY -> AGG : SPY exit event
#
# Direct QQQ <-> SPY is allowed ONLY when, on the SAME signal day:
#   QQQ -> SPY:
#       QQQ exit event + SPY entry event
#   SPY -> QQQ:
#       SPY exit event + QQQ entry event
#
# If the other condition was already true before today but did not
# just trigger an entry cross today, direct switching is NOT allowed.
#
# Execution:
#   Signal at today's CLOSE
#   -> execute at NEXT trading day's OPEN
#
# Transaction cost:
#   0
#
# Backtest reporting:
#   From first trading day of 2016
#
# Warm-up/state construction:
#   The state machine starts from CASH on the earliest valid signal date.
#   It then runs from 2010 historical data so that the state on
#   2016-01-01 is established by several years of prior cross events.
#   CASH is only the seed state; after the first valid entry, normal
#   QQQ / SPY / AGG transitions apply and exits go to AGG.
#
# Latest market data priority is handled in market_data.py:
#   Daily
#   -> 5m regular-session fallback
#   -> 1m regular-session fallback
#   -> fail safely
# ============================================================


DOWNLOAD_START = "2010-01-01"
BACKTEST_START = pd.Timestamp("2016-01-01")

NDX = "^NDX"
SPX = "^GSPC"

ASSETS = [
    "QQQ",
    "SPY",
    "AGG",
]

DEFENSIVE_ASSET = "AGG"
INITIAL_HOLDING = "CASH"
OUTPUT_DIR = Path("output")


# ============================================================
# INDICATORS + CROSS EVENTS
# ============================================================

def build_signals(ndx, spx):
    """
    Build indicator levels and one-day cross events.

    Important:
        This function does NOT decide the portfolio holding.
        Portfolio decisions are made later by the state machine.
    """

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
        .rolling(
            window=30,
            min_periods=30,
        )
        .mean()
    )

    signals["SPX MA50"] = (
        signals["SPX Close"]
        .rolling(
            window=50,
            min_periods=50,
        )
        .mean()
    )

    signals["SPX MA200"] = (
        signals["SPX Close"]
        .rolling(
            window=200,
            min_periods=200,
        )
        .mean()
    )

    signals = signals.dropna().copy()

    # --------------------------------------------------------
    # Current level conditions
    # --------------------------------------------------------

    signals["NDX > MA30"] = (
        signals["NDX Close"]
        > signals["NDX MA30"]
    )

    signals["SPX MA50 > MA200"] = (
        signals["SPX MA50"]
        > signals["SPX MA200"]
    )

    # --------------------------------------------------------
    # Cross events
    #
    # We intentionally do NOT count the first valid MA row as
    # a cross because there is no previous comparable row.
    # --------------------------------------------------------

    previous_ndx_above = (
        signals["NDX > MA30"]
        .shift(1)
    )

    previous_spx_above = (
        signals["SPX MA50 > MA200"]
        .shift(1)
    )

    signals["NDX Cross Up"] = (
        signals["NDX > MA30"]
        & previous_ndx_above.eq(False)
    )

    signals["NDX Cross Down"] = (
        (~signals["NDX > MA30"])
        & previous_ndx_above.eq(True)
    )

    signals["SPX Cross Up"] = (
        signals["SPX MA50 > MA200"]
        & previous_spx_above.eq(False)
    )

    signals["SPX Cross Down"] = (
        (~signals["SPX MA50 > MA200"])
        & previous_spx_above.eq(True)
    )

    return signals


# ============================================================
# EVENT-DRIVEN STATE MACHINE
# ============================================================

def apply_state_machine(
    signals,
    initial_holding=INITIAL_HOLDING,
):
    """
    Convert cross events into portfolio decisions.

    'Current Holding' means the asset held at today's close.
    'Signal' means the target to execute at next trading day's open.

    Direct QQQ <-> SPY switching is only allowed when the current
    holding exits AND the other holding triggers a fresh entry on
    the exact same signal day.

    The earliest state is CASH. CASH waits for the first valid entry
    event and is never used as the normal defensive state; exits from
    QQQ/SPY go to AGG.
    """

    result = signals.copy()

    current_holding = initial_holding

    current_holdings = []
    next_holdings = []
    reasons = []

    for _, row in result.iterrows():
        holding_at_close = current_holding

        ndx_above = bool(
            row["NDX > MA30"]
        )

        ndx_cross_up = bool(
            row["NDX Cross Up"]
        )

        ndx_cross_down = bool(
            row["NDX Cross Down"]
        )

        spx_cross_up = bool(
            row["SPX Cross Up"]
        )

        spx_cross_down = bool(
            row["SPX Cross Down"]
        )

        next_holding = holding_at_close

        # ====================================================
        # CURRENTLY QQQ
        # ====================================================

        if holding_at_close == "QQQ":

            if ndx_cross_down:

                # QQQ exits today. SPY may replace it directly
                # only if SPY also produces a fresh entry today.
                if spx_cross_up:
                    next_holding = "SPY"
                    reason = (
                        "NDX crossed below MA30 and "
                        "SPX MA50 crossed above MA200 "
                        "on the same day → QQQ → SPY"
                    )

                else:
                    next_holding = DEFENSIVE_ASSET
                    reason = (
                        "NDX crossed below MA30 "
                        f"→ QQQ exit → {DEFENSIVE_ASSET}"
                    )

            else:
                next_holding = "QQQ"
                reason = (
                    "Hold QQQ — no NDX down-cross below MA30"
                )

        # ====================================================
        # CURRENTLY SPY
        # ====================================================

        elif holding_at_close == "SPY":

            if spx_cross_down:

                # SPY exits today. QQQ may replace it directly
                # only if QQQ also produces a fresh entry today.
                if ndx_cross_up:
                    next_holding = "QQQ"
                    reason = (
                        "SPX MA50 crossed below MA200 and "
                        "NDX crossed above MA30 "
                        "on the same day → SPY → QQQ"
                    )

                else:
                    next_holding = DEFENSIVE_ASSET
                    reason = (
                        "SPX MA50 crossed below MA200 "
                        f"→ SPY exit → {DEFENSIVE_ASSET}"
                    )

            else:
                next_holding = "SPY"
                reason = (
                    "Hold SPY — no SPX MA50/MA200 down-cross"
                )

        # ====================================================
        # INITIAL SEED STATE: CASH
        # ====================================================

        elif holding_at_close == INITIAL_HOLDING:

            # CASH is only the seed state. It waits for a fresh, valid
            # entry event. Once the strategy leaves CASH, exits go to AGG
            # and the state machine never intentionally returns to CASH.
            if ndx_cross_up:
                next_holding = "QQQ"
                reason = (
                    "NDX crossed above MA30 → initial QQQ entry"
                )

            elif (
                spx_cross_up
                and not ndx_above
            ):
                next_holding = "SPY"
                reason = (
                    "SPX MA50 crossed above MA200 and "
                    "NDX <= MA30 → initial SPY entry"
                )

            else:
                next_holding = INITIAL_HOLDING
                reason = (
                    "No new valid initial entry cross → stay CASH"
                )

        # ====================================================
        # CURRENTLY DEFENSIVE ASSET (AGG)
        # ====================================================

        elif holding_at_close == DEFENSIVE_ASSET:

            # Rule 1 has priority.
            if ndx_cross_up:
                next_holding = "QQQ"
                reason = (
                    "NDX crossed above MA30 → QQQ entry"
                )

            # SPY entry requires a fresh golden cross AND
            # Rule 1 not satisfied on this day.
            elif (
                spx_cross_up
                and not ndx_above
            ):
                next_holding = "SPY"
                reason = (
                    "SPX MA50 crossed above MA200 and "
                    "NDX <= MA30 → SPY entry"
                )

            else:
                next_holding = DEFENSIVE_ASSET

                if (
                    spx_cross_up
                    and ndx_above
                ):
                    reason = (
                        "SPX MA50 crossed above MA200, but "
                        "Rule 1 is already satisfied "
                        f"(NDX > MA30) → stay {DEFENSIVE_ASSET}"
                    )
                else:
                    reason = (
                        "No new valid entry cross "
                        f"→ stay {DEFENSIVE_ASSET}"
                    )

        else:
            raise RuntimeError(
                f"Unknown holding state: {holding_at_close}"
            )

        current_holdings.append(
            holding_at_close
        )

        next_holdings.append(
            next_holding
        )

        reasons.append(
            reason
        )

        # The decision made at today's close becomes the holding
        # after the next trading day's open. Therefore it is the
        # current state when we evaluate the next close.
        current_holding = next_holding

    result["Current Holding"] = (
        current_holdings
    )

    result["Signal"] = (
        next_holdings
    )

    result["Reason"] = (
        reasons
    )

    return result


# ============================================================
# STATE-MACHINE VALIDATION
# ============================================================

def validate_state_machine(
    decisions,
    position_history,
):
    """
    Hard validation for the user's transition rule.

    QQQ -> SPY is legal only if:
        NDX Cross Down == True
        SPX Cross Up   == True

    SPY -> QQQ is legal only if:
        SPX Cross Down == True
        NDX Cross Up   == True
    """

    qqq_to_spy = 0
    spy_to_qqq = 0

    for _, change in position_history.iterrows():
        old_holding = change["From"]
        new_holding = change["To"]

        # Initial CASH allocation is not a state-to-state switch.
        if old_holding == "CASH":
            continue

        signal_date = pd.Timestamp(
            change["Signal Date"]
        )

        if signal_date not in decisions.index:
            raise RuntimeError(
                "Position History signal date is missing "
                f"from decision table: {signal_date.date()}"
            )

        row = decisions.loc[
            signal_date
        ]

        if (
            old_holding == "QQQ"
            and new_holding == "SPY"
        ):
            qqq_to_spy += 1

            if not (
                bool(row["NDX Cross Down"])
                and bool(row["SPX Cross Up"])
            ):
                raise RuntimeError(
                    "Illegal QQQ -> SPY switch on "
                    f"{signal_date.date()}. "
                    "Direct switching requires same-day "
                    "QQQ exit + SPY entry."
                )

        elif (
            old_holding == "SPY"
            and new_holding == "QQQ"
        ):
            spy_to_qqq += 1

            if not (
                bool(row["SPX Cross Down"])
                and bool(row["NDX Cross Up"])
            ):
                raise RuntimeError(
                    "Illegal SPY -> QQQ switch on "
                    f"{signal_date.date()}. "
                    "Direct switching requires same-day "
                    "SPY exit + QQQ entry."
                )

        elif (
            old_holding == "QQQ"
            and new_holding not in [
                "QQQ",
                DEFENSIVE_ASSET,
            ]
        ):
            raise RuntimeError(
                f"Illegal QQQ transition: "
                f"{old_holding} -> {new_holding}"
            )

        elif (
            old_holding == "SPY"
            and new_holding not in [
                "SPY",
                DEFENSIVE_ASSET,
            ]
        ):
            raise RuntimeError(
                f"Illegal SPY transition: "
                f"{old_holding} -> {new_holding}"
            )

        elif (
            old_holding == DEFENSIVE_ASSET
            and new_holding not in [
                DEFENSIVE_ASSET,
                "QQQ",
                "SPY",
            ]
        ):
            raise RuntimeError(
                f"Illegal {DEFENSIVE_ASSET} transition: "
                f"{old_holding} -> {new_holding}"
            )

    return {
        "QQQ -> SPY": qqq_to_spy,
        "SPY -> QQQ": spy_to_qqq,
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    signals,
    prices,
):
    """
    Signal is generated at today's CLOSE.

    The new position is entered at NEXT trading day's OPEN.

    Previous position receives:
        previous close -> next open

    New/current position receives:
        next open -> next close

    State-machine decisions are calculated on the complete common
    historical index BEFORE the 2016 reporting cutoff. This keeps
    the 2016 starting state dependent on pre-2016 cross events.
    """

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

    if len(common_index) < 2:
        raise RuntimeError(
            "Not enough common trading dates "
            "to run Strategy 1."
        )

    # --------------------------------------------------------
    # Apply the event-driven state machine on the FULL warm-up
    # history, not just from 2016 onward.
    # --------------------------------------------------------

    decisions = apply_state_machine(
        signals.loc[
            common_index
        ].copy()
    )

    execution = pd.DataFrame(
        index=common_index
    )

    # Today's open executes yesterday close's decision.
    execution["Target"] = (
        decisions["Signal"]
        .shift(1)
    )

    execution["Reason"] = (
        decisions["Reason"]
        .shift(1)
    )

    execution["Signal Date"] = (
        pd.Series(
            common_index,
            index=common_index,
        )
        .shift(1)
    )

    execution = execution.loc[
        execution.index
        >= BACKTEST_START
    ].copy()

    execution = execution.dropna(
        subset=[
            "Target",
            "Signal Date",
        ]
    )

    if execution.empty:
        raise RuntimeError(
            "No backtest dates available after 2016."
        )

    # --------------------------------------------------------
    # Internal consistency:
    # The position executed at today's open must equal the state
    # that the state machine considers current at today's close.
    # --------------------------------------------------------

    expected_current = (
        decisions.loc[
            execution.index,
            "Current Holding",
        ]
    )

    mismatches = (
        execution["Target"]
        != expected_current
    )

    if mismatches.any():
        bad_date = mismatches[
            mismatches
        ].index[0]

        raise RuntimeError(
            "State-machine/execution mismatch on "
            f"{bad_date.date()}."
        )

    equity = 1.0

    holding = None
    previous_date = None

    daily_rows = []
    position_history = []

    for date, row in execution.iterrows():

        target = row["Target"]

        signal_date = pd.Timestamp(
            row["Signal Date"]
        )

        reason = row["Reason"]

        # ====================================================
        # FIRST TRADING DAY OF REPORTED BACKTEST
        # ====================================================

        if previous_date is None:

            old_holding = INITIAL_HOLDING
            holding = target

            # CASH has a zero return. This is mainly a safety path in
            # case the warm-up period has not produced an entry by the
            # first reported backtest date.
            if holding != INITIAL_HOLDING:
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

                equity *= (
                    close_price
                    / open_price
                )

                position_history.append(
                    {
                        "Change": (
                            f"{old_holding} → {holding}"
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
                        "To": (
                            holding
                        ),
                        "Reason": (
                            reason
                        ),
                    }
                )

        # ====================================================
        # LATER TRADING DAYS
        # ====================================================

        else:

            # ------------------------------------------------
            # Overnight return belongs to OLD position.
            # ------------------------------------------------

            if holding != INITIAL_HOLDING:
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

            # ------------------------------------------------
            # Switch at today's open if yesterday's signal changed.
            # ------------------------------------------------

            if target != holding:

                old_holding = holding
                holding = target

                position_history.append(
                    {
                        "Change": (
                            f"{old_holding} → {holding}"
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
                        "To": (
                            holding
                        ),
                        "Reason": (
                            reason
                        ),
                    }
                )

            # ------------------------------------------------
            # Intraday return belongs to NEW/current position.
            # ------------------------------------------------

            if holding != INITIAL_HOLDING:
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

        decision_today = decisions.loc[
            date
        ]

        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Holding": holding,
                "Target": target,
                "Signal Date": signal_date,
                "Signal Reason": reason,
                "Today's Signal": (
                    decision_today["Signal"]
                ),
                "Today's Reason": (
                    decision_today["Reason"]
                ),
                "NDX Close": (
                    decision_today["NDX Close"]
                ),
                "NDX MA30": (
                    decision_today["NDX MA30"]
                ),
                "NDX > MA30": (
                    decision_today["NDX > MA30"]
                ),
                "NDX Cross Up": (
                    decision_today["NDX Cross Up"]
                ),
                "NDX Cross Down": (
                    decision_today["NDX Cross Down"]
                ),
                "SPX MA50": (
                    decision_today["SPX MA50"]
                ),
                "SPX MA200": (
                    decision_today["SPX MA200"]
                ),
                "SPX MA50 > MA200": (
                    decision_today["SPX MA50 > MA200"]
                ),
                "SPX Cross Up": (
                    decision_today["SPX Cross Up"]
                ),
                "SPX Cross Down": (
                    decision_today["SPX Cross Down"]
                ),
            }
        )

        previous_date = date

    daily = pd.DataFrame(
        daily_rows
    ).set_index(
        "Date"
    )

    history = pd.DataFrame(
        position_history
    )

    validation = validate_state_machine(
        decisions,
        history,
    )

    return (
        daily,
        history,
        decisions,
        validation,
    )


# ============================================================
# BENCHMARKS
# ============================================================

def buy_and_hold_equity(
    price_df,
    dates,
):
    """
    Benchmark begins at the OPEN of the first
    Strategy 1 reported backtest day.
    """

    first_date = dates[0]

    initial_open = (
        price_df.at[
            first_date,
            "Open",
        ]
    )

    equity = (
        price_df.loc[
            dates,
            "Close",
        ]
        / initial_open
    )

    return equity


# ============================================================
# MONTHLY / ANNUAL RETURNS
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
            f"Unknown period: {period}"
        )

    returns = (
        end_values.pct_change()
    )

    # First reporting period starts from equity = 1.
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

    prior_year_values = equity[
        equity.index.year
        < latest_year
    ]

    if len(
        prior_year_values
    ) > 0:

        ytd_base = (
            prior_year_values
            .iloc[-1]
        )

    else:

        ytd_base = 1.0

    ytd_return = (
        equity.iloc[-1]
        / ytd_base
    ) - 1.0

    since_2016 = (
        equity.iloc[-1]
        - 1.0
    )

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
            ** (
                1 / years
            )
        ) - 1.0

    else:

        cagr = 0.0

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


def percent(
    value,
):
    return (
        f"{value * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Strategy 1 — 双轨趋势（交叉事件驱动）"
    )
    print("=" * 70)

    # ========================================================
    # LATEST COMPLETED SESSION
    # ========================================================

    session = (
        get_latest_completed_session()
    )

    target_date = (
        session["date"]
    )

    print()
    print(
        "Latest completed market session:",
        target_date.date(),
    )
    print()

    # ========================================================
    # MARKET DATA
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
        price_sources[ticker] = source

    # ========================================================
    # INDICATORS + CROSS EVENTS
    # ========================================================

    signals = build_signals(
        ndx,
        spx,
    )

    # ========================================================
    # BACKTEST + STATE MACHINE
    # ========================================================

    (
        strategy_daily,
        position_history,
        decisions,
        validation,
    ) = run_backtest(
        signals,
        prices,
    )

    dates = (
        strategy_daily.index
    )

    # ========================================================
    # FRESHNESS CHECK
    # ========================================================

    if dates[-1] != target_date:

        raise RuntimeError(
            "Backtest did not reach the latest completed "
            "market session. "
            f"Expected {target_date.date()}, "
            f"got {dates[-1].date()}."
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
        monthly.index.astype(
            str
        )
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
    # SUMMARY
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

    latest_date = dates[-1]

    current_holding = (
        strategy_daily
        .iloc[-1]["Holding"]
    )

    latest_decision = (
        decisions.loc[
            latest_date
        ]
    )

    today_signal = (
        latest_decision["Signal"]
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
            latest_decision[
                "NDX Close"
            ]
        ),
        "NDX MA30": float(
            latest_decision[
                "NDX MA30"
            ]
        ),
        "NDX > MA30": bool(
            latest_decision[
                "NDX > MA30"
            ]
        ),
        "NDX Cross Up": bool(
            latest_decision[
                "NDX Cross Up"
            ]
        ),
        "NDX Cross Down": bool(
            latest_decision[
                "NDX Cross Down"
            ]
        ),
        "SPX MA50": float(
            latest_decision[
                "SPX MA50"
            ]
        ),
        "SPX MA200": float(
            latest_decision[
                "SPX MA200"
            ]
        ),
        "SPX MA50 > MA200": bool(
            latest_decision[
                "SPX MA50 > MA200"
            ]
        ),
        "SPX Cross Up": bool(
            latest_decision[
                "SPX Cross Up"
            ]
        ),
        "SPX Cross Down": bool(
            latest_decision[
                "SPX Cross Down"
            ]
        ),
        "Reason": (
            latest_decision[
                "Reason"
            ]
        ),
        "Data Sources": {
            "NDX": ndx_source,
            "SPX": spx_source,
            **price_sources,
        },
    }

    # ========================================================
    # SAVE OUTPUTS
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

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("BACKTEST PERIOD")
    print(
        f"{dates[0].date()}"
        f" → "
        f"{dates[-1].date()}"
    )

    print()
    print("=" * 70)
    print("DATA SOURCES")
    print("=" * 70)

    print(
        f"NDX : {ndx_source}"
    )
    print(
        f"SPX : {spx_source}"
    )

    for ticker in ASSETS:
        print(
            f"{ticker:<3} : "
            f"{price_sources[ticker]}"
        )

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
    print(
        "NDX Cross Up    :",
        current_status[
            "NDX Cross Up"
        ],
    )
    print(
        "NDX Cross Down  :",
        current_status[
            "NDX Cross Down"
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
    print(
        "SPX Cross Up    :",
        current_status[
            "SPX Cross Up"
        ],
    )
    print(
        "SPX Cross Down  :",
        current_status[
            "SPX Cross Down"
        ],
    )

    print()
    print(
        "Reason           :",
        current_status[
            "Reason"
        ],
    )

    print()
    print("=" * 70)
    print("STATE MACHINE VALIDATION")
    print("=" * 70)

    print(
        "Direct QQQ -> SPY switches :",
        validation[
            "QQQ -> SPY"
        ],
    )
    print(
        "Direct SPY -> QQQ switches :",
        validation[
            "SPY -> QQQ"
        ],
    )
    print(
        "✅ Every direct QQQ/SPY switch "
        "passed same-day exit + entry validation"
    )

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

    print()
    print(
        f"Position changes: "
        f"{len(position_history)}"
    )

    print()
    print(
        "✅ Strategy 1 event-driven "
        "backtest completed"
    )


if __name__ == "__main__":
    main()
