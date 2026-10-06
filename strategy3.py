from pathlib import Path
import json

import pandas as pd

from market_data import (
    MARKET_CALENDAR,
    get_latest_completed_session,
    load_daily_history,
)


# ============================================================
# Strategy 3 — QQQ / SPY 相对强弱（交叉事件驱动）
#
# State:
#   1 = QQQ
#   2 = SPY
#   3 = AGG
#   initial seed = CASH
#
# Daily signal timing:
#   Signal at today's CLOSE
#   -> execute at NEXT trading day's OPEN
#
# Relative-strength indicators:
#   N = QQQ daily Close / SPY daily Close
#
#   Weekly A = QQQ completed-week Close / SPY completed-week Close
#   V = average of latest 10 COMPLETED weekly A values
#
#   During an unfinished week, that week is NOT included in V.
#   On the week's final NYSE trading session, after the close, that
#   week becomes completed immediately, is included in V, and the
#   updated V is used for that same day's signal.
#
# Dynamic SPY 40-week MA:
#   MA40_today = (previous 39 completed weekly SPY closes
#                 + today's SPY Close) / 40
#
# Entry rules:
#   QQQ fresh entry:
#       yesterday N <= V
#       today     N >  V
#
#   SPY fresh entry:
#       yesterday SPY <= MA40
#       today     SPY >  MA40
#       AND today N <= V
#
# Holding rules:
#   QQQ holds while N > V.
#   SPY holds while N <= V AND SPY > MA40.
#
# Exit rules:
#   QQQ -> AGG when N <= V.
#   SPY -> AGG when N > V OR SPY <= MA40.
#
# Switching restriction:
#   QQQ <-> SPY direct switching is NEVER allowed.
#   Both sides must go through AGG.
#   A fresh entry event that occurs while exiting the other risk asset
#   is intentionally missed. AGG must wait for a FUTURE fresh crossing.
#
# Transaction cost:
#   0
#
# Backtest reporting:
#   From first trading day of 2016.
#
# Warm-up:
#   Download begins in 2010 so weekly indicators and portfolio state are
#   already established before the 2016 reporting window begins.
# ============================================================


DOWNLOAD_START = "2010-01-01"
BACKTEST_START = pd.Timestamp("2016-01-01")

ASSETS = ["QQQ", "SPY", "AGG"]
DEFENSIVE_ASSET = "AGG"
INITIAL_HOLDING = "CASH"
OUTPUT_DIR = Path("output")

WEEK_FREQ = "W-FRI"
V_WEEKS = 10
SPY_MA_WEEKS = 40


# ============================================================
# WEEKLY CALENDAR HELPERS
# ============================================================

def build_week_last_session_map(start_date, end_date):
    """
    Map each W-FRI week period to its actual final NYSE trading date.

    This is important for holiday-shortened weeks. For example, if Friday
    is a market holiday, Thursday can be the final completed session and
    the weekly close must become available immediately after Thursday's
    close.
    """

    schedule = MARKET_CALENDAR.schedule(
        start_date=pd.Timestamp(start_date).date(),
        end_date=(pd.Timestamp(end_date) + pd.Timedelta(days=7)).date(),
    )

    if schedule.empty:
        raise RuntimeError("No NYSE sessions available for weekly calendar.")

    session_dates = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()

    calendar = pd.DataFrame(index=session_dates)
    calendar["Week"] = calendar.index.to_period(WEEK_FREQ)

    return calendar.groupby("Week").apply(lambda x: x.index.max()).to_dict()


# ============================================================
# INDICATORS + CROSS EVENTS
# ============================================================

def build_signals(qqq, spy):
    """
    Build Strategy 3 daily indicators and fresh crossing events.

    V behavior:
      - unfinished week: use latest 10 completed weeks BEFORE current week
      - final trading session of week: include current week's completed
        weekly ratio immediately, then calculate that day's signal

    SPY MA40 behavior:
      - always use previous 39 completed weekly SPY closes
        + today's SPY daily close as the provisional/current weekly close
    """

    signals = pd.concat(
        [
            qqq["Close"].rename("QQQ Close"),
            spy["Close"].rename("SPY Close"),
        ],
        axis=1,
        join="inner",
    ).dropna().sort_index()

    if signals.empty:
        raise RuntimeError("No common QQQ/SPY history for Strategy 3.")

    signals["Week"] = signals.index.to_period(WEEK_FREQ)
    signals["N"] = signals["QQQ Close"] / signals["SPY Close"]

    # Weekly closes built from the common daily history. The current
    # unfinished week may appear here, but it is only included in V on the
    # official final NYSE session for that week.
    weekly = (
        signals.groupby("Week")[["QQQ Close", "SPY Close"]]
        .last()
        .copy()
    )
    weekly["Weekly Ratio"] = weekly["QQQ Close"] / weekly["SPY Close"]

    week_last_session = build_week_last_session_map(
        signals.index.min(),
        signals.index.max(),
    )

    v_values = []
    ma40_values = []
    week_complete_flags = []

    for date, row in signals.iterrows():
        week = row["Week"]

        official_last = week_last_session.get(week)
        week_complete_today = (
            official_last is not None
            and pd.Timestamp(date).normalize() == pd.Timestamp(official_last).normalize()
        )

        week_complete_flags.append(bool(week_complete_today))

        # ----------------------------------------------------
        # V = latest 10 completed weekly QQQ/SPY ratios.
        # On the final session of this week, current week is included.
        # Otherwise current week is excluded.
        # ----------------------------------------------------
        if week_complete_today:
            eligible_v = weekly.loc[weekly.index <= week, "Weekly Ratio"]
        else:
            eligible_v = weekly.loc[weekly.index < week, "Weekly Ratio"]

        if len(eligible_v) >= V_WEEKS:
            v = float(eligible_v.tail(V_WEEKS).mean())
        else:
            v = float("nan")

        v_values.append(v)

        # ----------------------------------------------------
        # Dynamic SPY 40-week MA:
        # previous 39 completed weekly closes + today's daily close.
        # Current week is NEVER counted inside those previous 39 weeks.
        # ----------------------------------------------------
        prior_spy_weeks = weekly.loc[weekly.index < week, "SPY Close"]

        if len(prior_spy_weeks) >= SPY_MA_WEEKS - 1:
            previous_39 = prior_spy_weeks.tail(SPY_MA_WEEKS - 1)
            ma40 = float(
                (previous_39.sum() + float(row["SPY Close"]))
                / SPY_MA_WEEKS
            )
        else:
            ma40 = float("nan")

        ma40_values.append(ma40)

    signals["V"] = v_values
    signals["SPY 40W MA"] = ma40_values
    signals["Week Complete"] = week_complete_flags

    # Both indicators must be valid before the state machine can run.
    signals = signals.dropna(subset=["V", "SPY 40W MA"]).copy()

    if signals.empty:
        raise RuntimeError("Not enough warm-up data for Strategy 3 indicators.")

    # --------------------------------------------------------
    # Current level conditions
    # --------------------------------------------------------
    signals["N > V"] = signals["N"] > signals["V"]
    signals["SPY > 40W MA"] = signals["SPY Close"] > signals["SPY 40W MA"]

    # --------------------------------------------------------
    # Fresh crossing events.
    #
    # Important: N/V crossing compares yesterday's actual N vs V with
    # today's actual N vs V. Therefore an end-of-week V update itself can
    # legitimately create a fresh crossing, exactly as agreed.
    # --------------------------------------------------------
    previous_n_above = signals["N > V"].shift(1)
    previous_spy_above = signals["SPY > 40W MA"].shift(1)

    signals["N Cross Up"] = signals["N > V"] & previous_n_above.eq(False)
    signals["N Cross Down"] = (~signals["N > V"]) & previous_n_above.eq(True)

    signals["SPY Cross Up"] = (
        signals["SPY > 40W MA"] & previous_spy_above.eq(False)
    )
    signals["SPY Cross Down"] = (
        (~signals["SPY > 40W MA"]) & previous_spy_above.eq(True)
    )

    return signals


# ============================================================
# EVENT-DRIVEN STATE MACHINE
# ============================================================

def apply_state_machine(signals, initial_holding=INITIAL_HOLDING):
    """
    Convert daily close events into next-open portfolio decisions.

    QQQ fresh entry:
        N fresh crosses from <= V to > V.

    QQQ hold:
        N > V.

    QQQ exit:
        N <= V -> AGG.

    SPY fresh entry:
        SPY fresh crosses from <= MA40 to > MA40
        AND N <= V.

    SPY hold:
        N <= V AND SPY > MA40.

    SPY exit:
        N > V OR SPY <= MA40 -> AGG.

    Direct QQQ <-> SPY switching is forbidden in both directions.
    """

    result = signals.copy()
    current_holding = initial_holding

    current_holdings = []
    next_holdings = []
    reasons = []

    for _, row in result.iterrows():
        holding_at_close = current_holding

        n_above = bool(row["N > V"])
        n_cross_up = bool(row["N Cross Up"])
        n_cross_down = bool(row["N Cross Down"])

        spy_above = bool(row["SPY > 40W MA"])
        spy_cross_up = bool(row["SPY Cross Up"])

        spy_ok = (not n_above) and spy_above
        spy_entry_event = spy_cross_up and (not n_above)

        next_holding = holding_at_close

        # ====================================================
        # CURRENTLY QQQ
        # ====================================================
        if holding_at_close == "QQQ":
            if n_above:
                next_holding = "QQQ"
                reason = "Hold QQQ — N remains above V"
            else:
                # No direct QQQ -> SPY, even if SPY has a fresh entry today.
                next_holding = DEFENSIVE_ASSET
                if spy_entry_event:
                    reason = (
                        "N <= V → QQQ exit → AGG; "
                        "same-day SPY fresh entry is intentionally missed "
                        "because direct QQQ → SPY is forbidden"
                    )
                elif n_cross_down:
                    reason = "N crossed to/below V → QQQ exit → AGG"
                else:
                    reason = "N <= V → QQQ exit → AGG"

        # ====================================================
        # CURRENTLY SPY
        # ====================================================
        elif holding_at_close == "SPY":
            if spy_ok:
                next_holding = "SPY"
                reason = "Hold SPY — N <= V AND SPY > 40W MA"
            else:
                # No direct SPY -> QQQ, even if N fresh-crosses up today.
                next_holding = DEFENSIVE_ASSET

                if n_above:
                    if n_cross_up:
                        reason = (
                            "N crossed above V → SPY condition invalid "
                            "→ SPY exit → AGG; QQQ fresh entry is intentionally "
                            "missed because direct SPY → QQQ is forbidden"
                        )
                    else:
                        reason = "SPY condition invalid: N > V → SPY exit → AGG"
                elif not spy_above:
                    reason = "SPY condition invalid: SPY <= 40W MA → SPY exit → AGG"
                else:
                    reason = "SPY condition became invalid → SPY exit → AGG"

        # ====================================================
        # INITIAL CASH
        # ====================================================
        elif holding_at_close == INITIAL_HOLDING:
            if n_cross_up:
                next_holding = "QQQ"
                reason = "N crossed above V → initial QQQ entry"
            elif spy_entry_event:
                next_holding = "SPY"
                reason = "SPY crossed above 40W MA while N <= V → initial SPY entry"
            else:
                next_holding = INITIAL_HOLDING
                reason = "No new valid initial entry cross → stay CASH"

        # ====================================================
        # CURRENTLY AGG
        # ====================================================
        elif holding_at_close == DEFENSIVE_ASSET:
            if n_cross_up:
                next_holding = "QQQ"
                reason = "N crossed above V → QQQ entry"
            elif spy_entry_event:
                next_holding = "SPY"
                reason = "SPY crossed above 40W MA while N <= V → SPY entry"
            else:
                next_holding = DEFENSIVE_ASSET
                reason = "No new valid entry cross → stay AGG"

        else:
            raise RuntimeError(f"Unknown holding state: {holding_at_close}")

        current_holdings.append(holding_at_close)
        next_holdings.append(next_holding)
        reasons.append(reason)

        # Today's close decision becomes the portfolio state after the next
        # session's open and therefore the state evaluated at next close.
        current_holding = next_holding

    result["Current Holding"] = current_holdings
    result["Signal"] = next_holdings
    result["Reason"] = reasons

    return result


# ============================================================
# STATE-MACHINE VALIDATION
# ============================================================

def validate_state_machine(decisions, position_history):
    """
    Hard validation for Strategy 3.

    Direct QQQ -> SPY and SPY -> QQQ are both forbidden.
    AGG/CASH entries require fresh crossing events.
    """

    qqq_to_spy = 0
    spy_to_qqq = 0

    for _, change in position_history.iterrows():
        old_holding = change["From"]
        new_holding = change["To"]
        signal_date = pd.Timestamp(change["Signal Date"])

        if signal_date not in decisions.index:
            raise RuntimeError(
                "Position History signal date missing from decisions: "
                f"{signal_date.date()}"
            )

        row = decisions.loc[signal_date]

        n_above = bool(row["N > V"])
        n_cross_up = bool(row["N Cross Up"])
        n_cross_down = bool(row["N Cross Down"])
        spy_above = bool(row["SPY > 40W MA"])
        spy_cross_up = bool(row["SPY Cross Up"])
        spy_entry_event = spy_cross_up and (not n_above)
        spy_ok = (not n_above) and spy_above

        if old_holding == "QQQ" and new_holding == "SPY":
            qqq_to_spy += 1
            raise RuntimeError(
                f"Forbidden QQQ -> SPY transition on {signal_date.date()}."
            )

        if old_holding == "SPY" and new_holding == "QQQ":
            spy_to_qqq += 1
            raise RuntimeError(
                f"Forbidden SPY -> QQQ transition on {signal_date.date()}."
            )

        if old_holding == INITIAL_HOLDING:
            if new_holding == "QQQ" and not n_cross_up:
                raise RuntimeError(
                    f"Illegal CASH -> QQQ on {signal_date.date()}: no N up-cross."
                )
            if new_holding == "SPY" and not spy_entry_event:
                raise RuntimeError(
                    f"Illegal CASH -> SPY on {signal_date.date()}: no valid SPY entry."
                )
            continue

        if old_holding == "QQQ":
            if new_holding == DEFENSIVE_ASSET and not n_cross_down:
                # In a logically consistent path this should be a fresh
                # down-cross because QQQ only holds while N > V.
                raise RuntimeError(
                    f"Illegal QQQ -> AGG on {signal_date.date()}: no N down-cross."
                )
            if new_holding not in ("QQQ", DEFENSIVE_ASSET):
                raise RuntimeError(f"Illegal QQQ transition: QQQ -> {new_holding}")

        elif old_holding == "SPY":
            if new_holding == DEFENSIVE_ASSET and spy_ok:
                raise RuntimeError(
                    f"Illegal SPY -> AGG on {signal_date.date()}: SPY condition still valid."
                )
            if new_holding not in ("SPY", DEFENSIVE_ASSET):
                raise RuntimeError(f"Illegal SPY transition: SPY -> {new_holding}")

        elif old_holding == DEFENSIVE_ASSET:
            if new_holding == "QQQ" and not n_cross_up:
                raise RuntimeError(
                    f"Illegal AGG -> QQQ on {signal_date.date()}: no N up-cross."
                )
            if new_holding == "SPY" and not spy_entry_event:
                raise RuntimeError(
                    f"Illegal AGG -> SPY on {signal_date.date()}: no valid SPY entry."
                )
            if new_holding not in (DEFENSIVE_ASSET, "QQQ", "SPY"):
                raise RuntimeError(f"Illegal AGG transition: AGG -> {new_holding}")

    return {
        "QQQ -> SPY": qqq_to_spy,
        "SPY -> QQQ": spy_to_qqq,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(signals, prices):
    """
    Signal is generated at today's close and executed next session open.

    Return mechanics on an execution day:
      old holding: previous close -> today's open
      new holding: today's open -> today's close

    State construction runs over the entire warm-up history before the
    2016 reporting cutoff so the first reported position is not fabricated.
    """

    common_index = signals.index.copy()

    for ticker in ASSETS:
        common_index = common_index.intersection(prices[ticker].index)

    common_index = common_index.sort_values()

    if len(common_index) < 2:
        raise RuntimeError("Not enough common trading dates to run Strategy 3.")

    decisions = apply_state_machine(signals.loc[common_index].copy())

    execution = pd.DataFrame(index=common_index)
    execution["Target"] = decisions["Signal"].shift(1)
    execution["Reason"] = decisions["Reason"].shift(1)
    execution["Signal Date"] = pd.Series(common_index, index=common_index).shift(1)

    execution = execution.loc[execution.index >= BACKTEST_START].copy()
    execution = execution.dropna(subset=["Target", "Signal Date"])

    if execution.empty:
        raise RuntimeError("No Strategy 3 backtest dates available after 2016.")

    # Position executed at today's open must equal state evaluated at
    # today's close before today's new signal is applied.
    expected_current = decisions.loc[execution.index, "Current Holding"]
    mismatch = execution["Target"] != expected_current

    if mismatch.any():
        bad_date = mismatch[mismatch].index[0]
        raise RuntimeError(
            "State-machine/execution mismatch on "
            f"{pd.Timestamp(bad_date).date()}."
        )

    equity = 1.0
    holding = None
    previous_date = None

    daily_rows = []
    position_history = []

    for date, row in execution.iterrows():
        target = row["Target"]
        signal_date = pd.Timestamp(row["Signal Date"])
        reason = row["Reason"]

        # ----------------------------------------------------
        # First reported trading day
        # ----------------------------------------------------
        if previous_date is None:
            old_holding = decisions.at[signal_date, "Current Holding"]
            holding = target

            if holding != old_holding:
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

            # Reporting begins at this first open. No overnight return from
            # before the reporting window is included.
            if holding != INITIAL_HOLDING:
                equity *= (
                    prices[holding].at[date, "Close"]
                    / prices[holding].at[date, "Open"]
                )

        # ----------------------------------------------------
        # Later trading days
        # ----------------------------------------------------
        else:
            # Overnight belongs to old holding.
            if holding != INITIAL_HOLDING:
                equity *= (
                    prices[holding].at[date, "Open"]
                    / prices[holding].at[previous_date, "Close"]
                )

            # Execute yesterday's signal at today's open.
            if target != holding:
                old_holding = holding
                holding = target

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

            # Intraday belongs to new/current holding.
            if holding != INITIAL_HOLDING:
                equity *= (
                    prices[holding].at[date, "Close"]
                    / prices[holding].at[date, "Open"]
                )

        decision_today = decisions.loc[date]

        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Holding": holding,
                "Target": target,
                "Signal Date": signal_date,
                "Signal Reason": reason,
                "Today's Signal": decision_today["Signal"],
                "Today's Reason": decision_today["Reason"],
                "QQQ Close": decision_today["QQQ Close"],
                "SPY Close": decision_today["SPY Close"],
                "N": decision_today["N"],
                "V": decision_today["V"],
                "N > V": decision_today["N > V"],
                "N Cross Up": decision_today["N Cross Up"],
                "N Cross Down": decision_today["N Cross Down"],
                "SPY 40W MA": decision_today["SPY 40W MA"],
                "SPY > 40W MA": decision_today["SPY > 40W MA"],
                "SPY Cross Up": decision_today["SPY Cross Up"],
                "SPY Cross Down": decision_today["SPY Cross Down"],
                "Week Complete": decision_today["Week Complete"],
            }
        )

        previous_date = date

    daily = pd.DataFrame(daily_rows).set_index("Date")

    history_columns = [
        "Change",
        "Signal Date",
        "Execute Date",
        "From",
        "To",
        "Reason",
    ]
    history = pd.DataFrame(position_history, columns=history_columns)

    validation = validate_state_machine(decisions, history)

    return daily, history, decisions, validation


# ============================================================
# BENCHMARKS / REPORTING HELPERS
# ============================================================

def buy_and_hold_equity(price_df, dates):
    first_date = dates[0]
    initial_open = price_df.at[first_date, "Open"]
    return price_df.loc[dates, "Close"] / initial_open


def calculate_period_returns(equity, period):
    if period == "month":
        end_values = equity.groupby(equity.index.to_period("M")).last()
    elif period == "year":
        end_values = equity.groupby(equity.index.year).last()
    else:
        raise ValueError(f"Unknown period: {period}")

    returns = end_values.pct_change()
    returns.iloc[0] = end_values.iloc[0] - 1.0
    return returns


def calculate_metrics(equity):
    latest_year = equity.index[-1].year
    prior_year_values = equity[equity.index.year < latest_year]
    ytd_base = prior_year_values.iloc[-1] if len(prior_year_values) > 0 else 1.0

    ytd_return = equity.iloc[-1] / ytd_base - 1.0
    since_2016 = equity.iloc[-1] - 1.0

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0

    drawdown = equity / equity.cummax() - 1.0

    return {
        "YTD Return": float(ytd_return),
        "Since 2016": float(since_2016),
        "CAGR": float(cagr),
        "Max Drawdown": float(drawdown.min()),
    }


def percent(value):
    return f"{value * 100:.2f}%"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Strategy 3 — QQQ / SPY 相对强弱（交叉事件驱动）")
    print("=" * 70)

    session = get_latest_completed_session()
    target_date = session["date"]

    print()
    print("Latest completed market session:", target_date.date())
    print()

    prices = {}
    price_sources = {}

    for ticker in ASSETS:
        df, source, _ = load_daily_history(
            ticker,
            DOWNLOAD_START,
            session=session,
        )
        prices[ticker] = df
        price_sources[ticker] = source

    signals = build_signals(prices["QQQ"], prices["SPY"])

    strategy_daily, position_history, decisions, validation = run_backtest(
        signals,
        prices,
    )

    dates = strategy_daily.index

    if dates[-1] != target_date:
        raise RuntimeError(
            "Backtest did not reach latest completed market session. "
            f"Expected {target_date.date()}, got {dates[-1].date()}."
        )

    strategy_equity = strategy_daily["Equity"]
    qqq_equity = buy_and_hold_equity(prices["QQQ"], dates)
    spy_equity = buy_and_hold_equity(prices["SPY"], dates)

    monthly = pd.DataFrame(
        {
            "Strategy 3": calculate_period_returns(strategy_equity, "month"),
            "QQQ": calculate_period_returns(qqq_equity, "month"),
            "SPY": calculate_period_returns(spy_equity, "month"),
        }
    )
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "Month"

    annual = pd.DataFrame(
        {
            "Strategy 3": calculate_period_returns(strategy_equity, "year"),
            "QQQ": calculate_period_returns(qqq_equity, "year"),
            "SPY": calculate_period_returns(spy_equity, "year"),
        }
    )
    annual.index.name = "Year"

    summary = {
        "Strategy 3": calculate_metrics(strategy_equity),
        "QQQ": calculate_metrics(qqq_equity),
        "SPY": calculate_metrics(spy_equity),
    }

    latest_date = dates[-1]
    current_holding = strategy_daily.iloc[-1]["Holding"]
    latest_decision = decisions.loc[latest_date]
    today_signal = latest_decision["Signal"]
    next_holding = today_signal
    pending_change = current_holding != next_holding

    current_status = {
        "Data Date": str(latest_date.date()),
        "Signal Date": str(latest_date.date()),
        "Current Holding": current_holding,
        "Today's Signal": today_signal,
        "Next Holding": next_holding,
        "Pending Change": bool(pending_change),
        "QQQ Close": float(latest_decision["QQQ Close"]),
        "SPY Close": float(latest_decision["SPY Close"]),
        "N": float(latest_decision["N"]),
        "V": float(latest_decision["V"]),
        "N > V": bool(latest_decision["N > V"]),
        "N Cross Up": bool(latest_decision["N Cross Up"]),
        "N Cross Down": bool(latest_decision["N Cross Down"]),
        "SPY 40W MA": float(latest_decision["SPY 40W MA"]),
        "SPY > 40W MA": bool(latest_decision["SPY > 40W MA"]),
        "SPY Cross Up": bool(latest_decision["SPY Cross Up"]),
        "SPY Cross Down": bool(latest_decision["SPY Cross Down"]),
        "Week Complete": bool(latest_decision["Week Complete"]),
        "Reason": latest_decision["Reason"],
        "Data Sources": price_sources,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategy_daily.to_csv(OUTPUT_DIR / "strategy3_daily.csv")
    monthly.to_csv(OUTPUT_DIR / "strategy3_monthly.csv")
    annual.to_csv(OUTPUT_DIR / "strategy3_annual.csv")
    position_history.to_csv(OUTPUT_DIR / "strategy3_position_history.csv", index=False)

    with open(
        OUTPUT_DIR / "strategy3_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "current_status": current_status,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("BACKTEST PERIOD")
    print(f"{dates[0].date()} → {dates[-1].date()}")

    print()
    print("=" * 70)
    print("DATA SOURCES")
    print("=" * 70)
    for ticker in ASSETS:
        print(f"{ticker:<3} : {price_sources[ticker]}")

    print()
    print("=" * 70)
    print("CURRENT STATUS")
    print("=" * 70)
    print("Data Date       :", latest_date.date())
    print("Current Holding :", current_holding)
    print("Today's Signal  :", today_signal)
    print("Next Holding    :", next_holding)
    print("Pending Change  :", "YES" if pending_change else "NO")

    print()
    print(f"QQQ Close       : {current_status['QQQ Close']:.4f}")
    print(f"SPY Close       : {current_status['SPY Close']:.4f}")
    print(f"N = QQQ/SPY     : {current_status['N']:.6f}")
    print(f"V (10W)         : {current_status['V']:.6f}")
    print("N > V           :", current_status["N > V"])
    print("N Cross Up      :", current_status["N Cross Up"])
    print("N Cross Down    :", current_status["N Cross Down"])
    print("Week Complete   :", current_status["Week Complete"])

    print()
    print(f"SPY 40W MA      : {current_status['SPY 40W MA']:.4f}")
    print("SPY > 40W MA    :", current_status["SPY > 40W MA"])
    print("SPY Cross Up    :", current_status["SPY Cross Up"])
    print("SPY Cross Down  :", current_status["SPY Cross Down"])

    print()
    print("Reason           :", current_status["Reason"])

    print()
    print("=" * 70)
    print("STATE MACHINE VALIDATION")
    print("=" * 70)
    print("Forbidden QQQ -> SPY switches:", validation["QQQ -> SPY"])
    print("Forbidden SPY -> QQQ switches:", validation["SPY -> QQQ"])
    print("✅ QQQ <-> SPY direct switching is forbidden; both directions must go through AGG")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, metrics in summary.items():
        print()
        print(name)
        print("  YTD Return    :", percent(metrics["YTD Return"]))
        print("  Since 2016    :", percent(metrics["Since 2016"]))
        print("  CAGR          :", percent(metrics["CAGR"]))
        print("  Max Drawdown  :", percent(metrics["Max Drawdown"]))

    print()
    print("=" * 70)
    print("LATEST 12 MONTHS")
    print("=" * 70)
    print(
        monthly.tail(12).to_string(
            float_format=lambda x: f"{x * 100:.2f}%"
        )
    )

    print()
    print("=" * 70)
    print("ANNUAL RETURNS")
    print("=" * 70)
    print(
        annual.to_string(
            float_format=lambda x: f"{x * 100:.2f}%"
        )
    )

    print()
    print(f"Position changes: {len(position_history)}")

    print()
    print("✅ Strategy 3 event-driven backtest completed")


if __name__ == "__main__":
    main()
