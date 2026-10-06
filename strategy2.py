from pathlib import Path
import json

import pandas as pd

from market_data import get_latest_completed_session, load_daily_history
from macro_data import BLS_SERIES_ID, build_daily_unemployment_state


# ============================================================
# Strategy 2 — 双轨 + 失业率（事件驱动）
#
# 1 = QQQ
# 2 = SPY
# 3 = AGG
# 初始 seed = CASH
#
# QQQ:
#   入场：NDX Close 从 <= MA30 上穿到 > MA30
#   出场：NDX Close 从 > MA30 下穿到 <= MA30
#
# SPY:
#   A = SPX 当前月线 >= SPX 12个月移动平均线
#   B = 最新已公布失业率 <= 失业率12个月移动平均线
#   Macro Condition = A OR B
#   SPY Condition = (NDX <= MA30) AND Macro Condition
#
#   入场：Macro Condition False -> True
#         且当天 NDX <= MA30（条件1不满足）
#   持有：NDX <= MA30 AND Macro Condition = True
#   出场：SPY Condition 任一部分失效：
#         NDX > MA30 OR Macro Condition = False
#
# SPX 月线每天收盘后更新：
#   SPX Month Close = 当天 SPX Close
#   SPX 12M MA = 前11个完整月份月收盘 + 当前月截至当天收盘，共12个值平均
#
# 失业率按实际发布日期生效，不能提前看到未来数据。
#
# QQQ <-> SPY 只有在同一天“旧持仓刚退出 + 新持仓刚触发入场”时才允许直切。
# 其他情况退出后进入 AGG。
#
# 信号：当天收盘
# 执行：下一交易日开盘
# 交易成本：0
# 回测展示：2016年第一交易日起
# ============================================================

DOWNLOAD_START = "2008-01-01"
MACRO_START_YEAR = 2008
BACKTEST_START = pd.Timestamp("2016-01-01")

NDX = "^NDX"
SPX = "^GSPC"
ASSETS = ["QQQ", "SPY", "AGG"]
DEFENSIVE_ASSET = "AGG"
INITIAL_HOLDING = "CASH"
OUTPUT_DIR = Path("output")


# ============================================================
# SPX 月线：每天动态计算
# ============================================================

def build_spx_monthly_state(spx_close: pd.Series) -> pd.DataFrame:
    """
    对每个交易日计算：
      SPX Month Close = 当天收盘
      SPX 12M MA = 前11个完整月月末收盘 + 当天收盘，然后除以12
    """
    close = spx_close.dropna().copy()
    if close.empty:
        raise RuntimeError("SPX close history is empty.")

    close.index = pd.DatetimeIndex(close.index).normalize()
    month_key = close.index.to_period("M")

    # 每个月真正的月末收盘。历史某一天只会使用它之前的月份。
    completed_month_close = close.groupby(month_key).last()

    # 对月份 M 来说，这里正好是 M-1 ... M-11。
    previous_11 = completed_month_close.shift(1).rolling(window=11, min_periods=11)
    previous_11_sum = previous_11.sum()
    previous_11_count = previous_11.count()

    prior_sum_for_day = pd.Series(
        month_key.map(previous_11_sum), index=close.index, dtype="float64"
    )
    prior_count_for_day = pd.Series(
        month_key.map(previous_11_count), index=close.index, dtype="float64"
    )

    result = pd.DataFrame(index=close.index)
    result["SPX Month Close"] = close
    result["SPX 12M MA"] = (prior_sum_for_day + result["SPX Month Close"]) / 12.0
    result.loc[prior_count_for_day != 11, "SPX 12M MA"] = pd.NA
    result["SPX >= 12M MA"] = result["SPX Month Close"] >= result["SPX 12M MA"]
    return result


# ============================================================
# 指标 + 触发事件
# ============================================================

def build_signals(ndx, spx, unemployment_daily):
    """
    Build daily Strategy 2 states and fresh entry/exit events.

    QQQ:
        NDX > MA30

    Macro Condition:
        SPX current-month close >= dynamic 12M MA
        OR
        latest available unemployment rate <= its 12M MA

    SPY valid / hold condition:
        NDX <= MA30 AND Macro Condition

    Important:
        A new SPY entry is NOT created merely because the full SPY
        condition becomes true due to NDX falling below MA30.

        SPY entry requires a fresh Macro Condition False -> True event
        while NDX <= MA30 on that same signal day.
    """

    ndx_state = pd.DataFrame(
        {
            "NDX Close": ndx["Close"]
        }
    )

    ndx_state["NDX MA30"] = (
        ndx_state["NDX Close"]
        .rolling(
            30,
            min_periods=30,
        )
        .mean()
    )

    spx_monthly = build_spx_monthly_state(
        spx["Close"]
    )

    macro_columns = [
        "Reference Month",
        "Release Date",
        "Unemployment Rate",
        "Unemployment 12M MA",
        "Unemployment <= 12M MA",
        "Release Date Source",
    ]

    missing = [
        column
        for column in macro_columns
        if column not in unemployment_daily.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing unemployment columns: "
            + ", ".join(missing)
        )

    signals = pd.concat(
        [
            ndx_state,
            spx_monthly,
            unemployment_daily[macro_columns],
        ],
        axis=1,
        join="inner",
    )

    signals = signals.dropna(
        subset=[
            "NDX Close",
            "NDX MA30",
            "SPX Month Close",
            "SPX 12M MA",
            "Unemployment Rate",
            "Unemployment 12M MA",
        ]
    ).copy()

    if signals.empty:
        raise RuntimeError(
            "No common valid Strategy 2 signal dates."
        )

    # ========================================================
    # CURRENT STATES
    # ========================================================

    signals["NDX > MA30"] = (
        signals["NDX Close"]
        > signals["NDX MA30"]
    )

    signals["SPX >= 12M MA"] = (
        signals["SPX Month Close"]
        >= signals["SPX 12M MA"]
    )

    signals["Unemployment <= 12M MA"] = (
        signals["Unemployment Rate"]
        <= signals["Unemployment 12M MA"]
    )

    # Macro side of Strategy 2.
    signals["Macro Condition"] = (
        signals["SPX >= 12M MA"]
        | signals["Unemployment <= 12M MA"]
    )

    # Full SPY validity is an AND condition.
    signals["SPY Condition"] = (
        (~signals["NDX > MA30"])
        & signals["Macro Condition"]
    )

    # ========================================================
    # FRESH EVENTS
    # ========================================================

    previous_ndx_above = (
        signals["NDX > MA30"]
        .shift(1)
    )

    previous_macro_condition = (
        signals["Macro Condition"]
        .shift(1)
    )

    previous_spy_condition = (
        signals["SPY Condition"]
        .shift(1)
    )

    # First valid row is never treated as a cross.
    signals["NDX Cross Up"] = (
        signals["NDX > MA30"]
        & previous_ndx_above.eq(False)
    )

    signals["NDX Cross Down"] = (
        (~signals["NDX > MA30"])
        & previous_ndx_above.eq(True)
    )

    signals["Macro Cross Up"] = (
        signals["Macro Condition"]
        & previous_macro_condition.eq(False)
    )

    signals["Macro Cross Down"] = (
        (~signals["Macro Condition"])
        & previous_macro_condition.eq(True)
    )

    # SPY Cross Up means a FRESH VALID SPY ENTRY EVENT:
    # Macro Condition freshly turns True while NDX <= MA30.
    # A later NDX down-cross while Macro Condition was already True
    # does NOT create a new SPY entry event.
    signals["SPY Cross Up"] = (
        signals["Macro Cross Up"]
        & (~signals["NDX > MA30"])
    )

    # SPY Cross Down tracks the full AND condition becoming invalid.
    signals["SPY Cross Down"] = (
        (~signals["SPY Condition"])
        & previous_spy_condition.eq(True)
    )

    return signals


# ============================================================
# 状态机
# ============================================================

def apply_state_machine(
    signals,
    initial_holding=INITIAL_HOLDING,
):
    """
    Convert daily close events into next-open Strategy 2 decisions.

    QQQ entry:
        NDX fresh cross from <= MA30 to > MA30.

    QQQ exit:
        NDX fresh cross from > MA30 to <= MA30.

    Macro Condition:
        SPX >= dynamic 12M MA OR unemployment <= unemployment 12M MA.

    SPY valid / hold condition:
        NDX <= MA30 AND Macro Condition.

    SPY fresh entry:
        Macro Condition fresh crosses False -> True while NDX <= MA30.

    SPY exit:
        The full SPY AND condition becomes false for any reason:
        NDX > MA30 OR Macro Condition == False.

    Direct switching:
        QQQ -> SPY only when QQQ exits and SPY has a fresh valid
        entry event on the same signal day.

        SPY -> QQQ only when SPY becomes invalid and QQQ has a fresh
        NDX up-cross on the same signal day.

    CASH is only the seed state. After first leaving CASH, normal
    defensive exits go to AGG and the strategy never intentionally
    returns to CASH.
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

        macro_condition = bool(
            row["Macro Condition"]
        )

        macro_cross_up = bool(
            row["Macro Cross Up"]
        )

        spy_ok = bool(
            row["SPY Condition"]
        )

        # This is the only fresh SPY entry event.
        spy_entry_event = bool(
            row["SPY Cross Up"]
        )

        next_holding = holding_at_close

        # ====================================================
        # CURRENTLY QQQ
        # ====================================================

        if holding_at_close == "QQQ":

            if ndx_cross_down:

                # Direct QQQ -> SPY is allowed only when the SPY
                # macro trigger also freshly fires on this same day.
                if spy_entry_event:
                    next_holding = "SPY"
                    reason = (
                        "NDX crossed to/below MA30 and Macro Condition "
                        "crossed False→True on the same day while "
                        "NDX <= MA30 → QQQ → SPY"
                    )

                else:
                    next_holding = DEFENSIVE_ASSET
                    reason = (
                        "NDX crossed to/below MA30 "
                        f"→ QQQ exit → {DEFENSIVE_ASSET}"
                    )

            else:
                next_holding = "QQQ"
                reason = (
                    "Hold QQQ — NDX remains above MA30 "
                    "with no fresh down-cross"
                )

        # ====================================================
        # CURRENTLY SPY
        # ====================================================

        elif holding_at_close == "SPY":

            # SPY must continuously satisfy BOTH:
            #   NDX <= MA30
            #   Macro Condition == True
            # If either one fails, SPY exits.
            if not spy_ok:

                # If NDX fresh-crosses above MA30 today, SPY becomes
                # invalid and QQQ receives a fresh entry on the same
                # signal day, so direct SPY -> QQQ is legal.
                if ndx_cross_up:
                    next_holding = "QQQ"
                    reason = (
                        "SPY condition became invalid because NDX "
                        "crossed above MA30, and QQQ received a fresh "
                        "entry on the same day → SPY → QQQ"
                    )

                else:
                    next_holding = DEFENSIVE_ASSET

                    if ndx_above:
                        reason = (
                            "SPY condition invalid: NDX > MA30 "
                            f"→ SPY exit → {DEFENSIVE_ASSET}"
                        )
                    elif not macro_condition:
                        reason = (
                            "SPY condition invalid: Macro Condition is False "
                            f"→ SPY exit → {DEFENSIVE_ASSET}"
                        )
                    else:
                        reason = (
                            "SPY AND condition became invalid "
                            f"→ SPY exit → {DEFENSIVE_ASSET}"
                        )

            else:
                next_holding = "SPY"
                reason = (
                    "Hold SPY — NDX <= MA30 AND Macro Condition is True"
                )

        # ====================================================
        # INITIAL SEED STATE: CASH
        # ====================================================

        elif holding_at_close == INITIAL_HOLDING:

            # Rule 1 / QQQ has priority.
            if ndx_cross_up:
                next_holding = "QQQ"
                reason = (
                    "NDX crossed above MA30 → initial QQQ entry"
                )

            elif spy_entry_event:
                next_holding = "SPY"
                reason = (
                    "Macro Condition crossed False→True while "
                    "NDX <= MA30 → initial SPY entry"
                )

            else:
                next_holding = INITIAL_HOLDING

                if macro_cross_up and ndx_above:
                    reason = (
                        "Macro Condition crossed False→True, but "
                        "NDX > MA30 so the SPY AND condition is invalid "
                        "→ stay CASH"
                    )
                else:
                    reason = (
                        "No new valid initial entry event → stay CASH"
                    )

        # ====================================================
        # CURRENTLY DEFENSIVE ASSET (AGG)
        # ====================================================

        elif holding_at_close == DEFENSIVE_ASSET:

            # Rule 1 / QQQ has priority.
            if ndx_cross_up:
                next_holding = "QQQ"
                reason = (
                    "NDX crossed above MA30 → QQQ entry"
                )

            elif spy_entry_event:
                next_holding = "SPY"
                reason = (
                    "Macro Condition crossed False→True while "
                    "NDX <= MA30 → SPY entry"
                )

            else:
                next_holding = DEFENSIVE_ASSET

                if macro_cross_up and ndx_above:
                    reason = (
                        "Macro Condition crossed False→True, but "
                        "NDX > MA30 so the SPY AND condition is invalid "
                        f"→ stay {DEFENSIVE_ASSET}"
                    )
                elif spy_ok:
                    reason = (
                        "SPY condition is currently valid but there was "
                        "no fresh Macro False→True entry event today "
                        f"→ stay {DEFENSIVE_ASSET}"
                    )
                else:
                    reason = (
                        "No new valid entry event "
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

        current_holding = next_holding

    result["Current Holding"] = current_holdings
    result["Signal"] = next_holdings
    result["Reason"] = reasons

    return result


# ============================================================
# 状态机硬校验
# ============================================================

def validate_state_machine(
    decisions,
    position_history,
):
    """
    Hard validation for Strategy 2 transitions.

    QQQ -> SPY requires same-day:
        QQQ exit (NDX Cross Down)
        fresh valid SPY entry (Macro Cross Up while NDX <= MA30)

    SPY -> QQQ requires same-day:
        full SPY condition invalid
        fresh QQQ entry (NDX Cross Up)

    QQQ -> AGG requires a fresh QQQ exit.
    SPY -> AGG requires the full SPY AND condition to be invalid.
    AGG -> QQQ / SPY requires a fresh valid entry event.
    """

    qqq_to_spy = 0
    spy_to_qqq = 0

    for _, change in position_history.iterrows():
        old_holding = change["From"]
        new_holding = change["To"]

        signal_date = pd.Timestamp(
            change["Signal Date"]
        )

        if signal_date not in decisions.index:
            raise RuntimeError(
                "Position History signal date missing from decisions: "
                f"{signal_date.date()}"
            )

        row = decisions.loc[
            signal_date
        ]

        ndx_cross_up = bool(
            row["NDX Cross Up"]
        )

        ndx_cross_down = bool(
            row["NDX Cross Down"]
        )

        spy_ok = bool(
            row["SPY Condition"]
        )

        spy_entry_event = bool(
            row["SPY Cross Up"]
        )

        # ----------------------------------------------------
        # Initial CASH allocation
        # ----------------------------------------------------

        if old_holding == INITIAL_HOLDING:

            if new_holding == "QQQ":
                if not ndx_cross_up:
                    raise RuntimeError(
                        "Illegal CASH -> QQQ transition on "
                        f"{signal_date.date()}: no fresh NDX up-cross."
                    )

            elif new_holding == "SPY":
                if not spy_entry_event:
                    raise RuntimeError(
                        "Illegal CASH -> SPY transition on "
                        f"{signal_date.date()}: no fresh valid SPY entry."
                    )

            elif new_holding != INITIAL_HOLDING:
                raise RuntimeError(
                    f"Illegal CASH transition: CASH -> {new_holding}"
                )

            continue

        # ----------------------------------------------------
        # QQQ transitions
        # ----------------------------------------------------

        if old_holding == "QQQ":

            if new_holding == "SPY":
                qqq_to_spy += 1

                if not (
                    ndx_cross_down
                    and spy_entry_event
                ):
                    raise RuntimeError(
                        "Illegal QQQ -> SPY on "
                        f"{signal_date.date()}: direct switching "
                        "requires same-day QQQ exit + fresh valid SPY entry."
                    )

            elif new_holding == DEFENSIVE_ASSET:
                if not ndx_cross_down:
                    raise RuntimeError(
                        "Illegal QQQ -> AGG on "
                        f"{signal_date.date()}: no NDX down-cross."
                    )

            elif new_holding != "QQQ":
                raise RuntimeError(
                    f"Illegal QQQ transition: {old_holding} -> {new_holding}"
                )

        # ----------------------------------------------------
        # SPY transitions
        # ----------------------------------------------------

        elif old_holding == "SPY":

            if new_holding == "QQQ":
                spy_to_qqq += 1

                if not (
                    (not spy_ok)
                    and ndx_cross_up
                ):
                    raise RuntimeError(
                        "Illegal SPY -> QQQ on "
                        f"{signal_date.date()}: direct switching requires "
                        "same-day SPY invalidation + fresh QQQ entry."
                    )

            elif new_holding == DEFENSIVE_ASSET:
                if spy_ok:
                    raise RuntimeError(
                        "Illegal SPY -> AGG on "
                        f"{signal_date.date()}: SPY AND condition was still valid."
                    )

            elif new_holding != "SPY":
                raise RuntimeError(
                    f"Illegal SPY transition: {old_holding} -> {new_holding}"
                )

        # ----------------------------------------------------
        # AGG transitions
        # ----------------------------------------------------

        elif old_holding == DEFENSIVE_ASSET:

            if new_holding == "QQQ":
                if not ndx_cross_up:
                    raise RuntimeError(
                        "Illegal AGG -> QQQ on "
                        f"{signal_date.date()}: no fresh NDX up-cross."
                    )

            elif new_holding == "SPY":
                if not spy_entry_event:
                    raise RuntimeError(
                        "Illegal AGG -> SPY on "
                        f"{signal_date.date()}: no fresh valid SPY entry."
                    )

            elif new_holding != DEFENSIVE_ASSET:
                raise RuntimeError(
                    f"Illegal AGG transition: {old_holding} -> {new_holding}"
                )

        else:
            raise RuntimeError(
                f"Unexpected prior holding: {old_holding}"
            )

    return {
        "QQQ -> SPY": qqq_to_spy,
        "SPY -> QQQ": spy_to_qqq,
    }


# ============================================================
# 收益计算辅助
# ============================================================

def apply_overnight_return(equity, holding, prices, previous_date, date):
    if holding == INITIAL_HOLDING:
        return equity
    return equity * (
        prices[holding].at[date, "Open"]
        / prices[holding].at[previous_date, "Close"]
    )


def apply_intraday_return(equity, holding, prices, date):
    if holding == INITIAL_HOLDING:
        return equity
    return equity * (
        prices[holding].at[date, "Close"]
        / prices[holding].at[date, "Open"]
    )


# ============================================================
# 回测
# ============================================================

def run_backtest(signals, prices):
    common_index = signals.index.copy()

    for ticker in ASSETS:
        common_index = common_index.intersection(
            prices[ticker].index
        )

    common_index = common_index.sort_values()

    if len(common_index) < 2:
        raise RuntimeError(
            "Not enough common trading dates to run Strategy 2."
        )

    # Run the event-driven state machine on the FULL warm-up
    # history before cutting the displayed backtest to 2016.
    decisions = apply_state_machine(
        signals.loc[common_index].copy()
    )

    execution = pd.DataFrame(
        index=common_index
    )

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
        execution.index >= BACKTEST_START
    ].copy()

    execution = execution.dropna(
        subset=[
            "Target",
            "Signal Date",
        ]
    )

    if execution.empty:
        raise RuntimeError(
            "No Strategy 2 backtest dates available after 2016."
        )

    expected_current = decisions.loc[
        execution.index,
        "Current Holding",
    ]

    mismatch = (
        execution["Target"]
        != expected_current
    )

    if mismatch.any():
        bad_date = mismatch[
            mismatch
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
        # FIRST REPORTED TRADING DAY
        # ====================================================

        if previous_date is None:

            # The strategy state already exists from pre-2016 warm-up.
            # Use that true prior holding rather than fabricating CASH -> X.
            old_holding = decisions.at[
                signal_date,
                "Current Holding",
            ]

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

            # Performance reporting starts at this day's OPEN.
            if holding != INITIAL_HOLDING:
                equity = apply_intraday_return(
                    equity,
                    holding,
                    prices,
                    date,
                )

        # ====================================================
        # LATER TRADING DAYS
        # ====================================================

        else:
            # Previous close -> today's open belongs to the OLD holding.
            equity = apply_overnight_return(
                equity,
                holding,
                prices,
                previous_date,
                date,
            )

            # Today's open executes yesterday close's signal.
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

            # Today's open -> close belongs to the NEW/current holding.
            equity = apply_intraday_return(
                equity,
                holding,
                prices,
                date,
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
                "Today's Signal": decision_today["Signal"],
                "Today's Reason": decision_today["Reason"],
                "NDX Close": decision_today["NDX Close"],
                "NDX MA30": decision_today["NDX MA30"],
                "NDX > MA30": decision_today["NDX > MA30"],
                "NDX Cross Up": decision_today["NDX Cross Up"],
                "NDX Cross Down": decision_today["NDX Cross Down"],
                "SPX Month Close": decision_today["SPX Month Close"],
                "SPX 12M MA": decision_today["SPX 12M MA"],
                "SPX >= 12M MA": decision_today["SPX >= 12M MA"],
                "Unemployment Reference Month": str(
                    decision_today["Reference Month"]
                ),
                "Unemployment Release Date": pd.Timestamp(
                    decision_today["Release Date"]
                ),
                "Unemployment Rate": decision_today["Unemployment Rate"],
                "Unemployment 12M MA": decision_today["Unemployment 12M MA"],
                "Unemployment <= 12M MA": decision_today[
                    "Unemployment <= 12M MA"
                ],
                "Macro Condition": decision_today["Macro Condition"],
                "Macro Cross Up": decision_today["Macro Cross Up"],
                "Macro Cross Down": decision_today["Macro Cross Down"],
                "SPY Condition": decision_today["SPY Condition"],
                "SPY Cross Up": decision_today["SPY Cross Up"],
                "SPY Cross Down": decision_today["SPY Cross Down"],
            }
        )

        previous_date = date

    daily = pd.DataFrame(
        daily_rows
    ).set_index(
        "Date"
    )

    history = pd.DataFrame(
        position_history,
        columns=[
            "Change",
            "Signal Date",
            "Execute Date",
            "From",
            "To",
            "Reason",
        ],
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
# Benchmark / 统计
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
    previous_year = equity[equity.index.year < latest_year]
    ytd_base = previous_year.iloc[-1] if len(previous_year) else 1.0
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


def date_string(value):
    return None if pd.isna(value) else str(pd.Timestamp(value).date())


def period_string(value):
    return None if pd.isna(value) else str(value)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Strategy 2 — 双轨 + 失业率（交叉事件驱动）")
    print("=" * 70)

    session = get_latest_completed_session()
    target_date = session["date"]
    print("\nLatest completed market session:", target_date.date())

    ndx, ndx_source, _ = load_daily_history(NDX, DOWNLOAD_START, session=session)
    spx, spx_source, _ = load_daily_history(SPX, DOWNLOAD_START, session=session)

    prices = {}
    price_sources = {}
    for ticker in ASSETS:
        df, source, _ = load_daily_history(ticker, DOWNLOAD_START, session=session)
        prices[ticker] = df
        price_sources[ticker] = source

    # 失业率映射到真实交易日，并严格按发布日期向后看。
    macro_dates = ndx.index.intersection(spx.index).sort_values()
    unemployment_daily = build_daily_unemployment_state(
        trading_dates=macro_dates,
        start_year=MACRO_START_YEAR,
    )

    signals = build_signals(ndx, spx, unemployment_daily)
    strategy_daily, position_history, decisions, validation = run_backtest(
        signals, prices
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
            "Strategy 2": calculate_period_returns(strategy_equity, "month"),
            "QQQ": calculate_period_returns(qqq_equity, "month"),
            "SPY": calculate_period_returns(spy_equity, "month"),
        }
    )
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "Month"

    annual = pd.DataFrame(
        {
            "Strategy 2": calculate_period_returns(strategy_equity, "year"),
            "QQQ": calculate_period_returns(qqq_equity, "year"),
            "SPY": calculate_period_returns(spy_equity, "year"),
        }
    )
    annual.index.name = "Year"

    summary = {
        "Strategy 2": calculate_metrics(strategy_equity),
        "QQQ": calculate_metrics(qqq_equity),
        "SPY": calculate_metrics(spy_equity),
    }

    latest_date = dates[-1]
    current_holding = strategy_daily.iloc[-1]["Holding"]
    latest = decisions.loc[latest_date]
    today_signal = latest["Signal"]
    next_holding = today_signal
    pending_change = current_holding != next_holding

    current_status = {
        "Data Date": str(latest_date.date()),
        "Signal Date": str(latest_date.date()),
        "Current Holding": current_holding,
        "Today's Signal": today_signal,
        "Next Holding": next_holding,
        "Pending Change": bool(pending_change),
        "NDX Close": float(latest["NDX Close"]),
        "NDX MA30": float(latest["NDX MA30"]),
        "NDX > MA30": bool(latest["NDX > MA30"]),
        "NDX Cross Up": bool(latest["NDX Cross Up"]),
        "NDX Cross Down": bool(latest["NDX Cross Down"]),
        "SPX Month Close": float(latest["SPX Month Close"]),
        "SPX 12M MA": float(latest["SPX 12M MA"]),
        "SPX >= 12M MA": bool(latest["SPX >= 12M MA"]),
        "Unemployment Reference Month": period_string(latest["Reference Month"]),
        "Unemployment Release Date": date_string(latest["Release Date"]),
        "Unemployment Rate": float(latest["Unemployment Rate"]),
        "Unemployment 12M MA": float(latest["Unemployment 12M MA"]),
        "Unemployment <= 12M MA": bool(latest["Unemployment <= 12M MA"]),
        "Macro Condition": bool(latest["Macro Condition"]),
        "Macro Cross Up": bool(latest["Macro Cross Up"]),
        "Macro Cross Down": bool(latest["Macro Cross Down"]),
        "SPY Condition": bool(latest["SPY Condition"]),
        "SPY Cross Up": bool(latest["SPY Cross Up"]),
        "SPY Cross Down": bool(latest["SPY Cross Down"]),
        "Reason": latest["Reason"],
        "Data Sources": {
            "NDX": ndx_source,
            "SPX": spx_source,
            **price_sources,
            "Unemployment": (
                f"U.S. Bureau of Labor Statistics {BLS_SERIES_ID}; "
                "Employment Situation release-date mapping"
            ),
            "Unemployment Release Date Source": str(latest["Release Date Source"]),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_daily.to_csv(OUTPUT_DIR / "strategy2_daily.csv")
    monthly.to_csv(OUTPUT_DIR / "strategy2_monthly.csv")
    annual.to_csv(OUTPUT_DIR / "strategy2_annual.csv")
    position_history.to_csv(
        OUTPUT_DIR / "strategy2_position_history.csv", index=False
    )
    with open(OUTPUT_DIR / "strategy2_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "current_status": current_status},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nBACKTEST PERIOD")
    print(f"{dates[0].date()} → {dates[-1].date()}")

    print("\n" + "=" * 70)
    print("DATA SOURCES")
    print("=" * 70)
    print(f"NDX : {ndx_source}")
    print(f"SPX : {spx_source}")
    for ticker in ASSETS:
        print(f"{ticker:<3} : {price_sources[ticker]}")
    print(f"BLS : {BLS_SERIES_ID} + Employment Situation release dates")

    print("\n" + "=" * 70)
    print("CURRENT STATUS")
    print("=" * 70)
    print("Data Date          :", latest_date.date())
    print("Current Holding    :", current_holding)
    print("Today's Signal     :", today_signal)
    print("Next Holding       :", next_holding)
    print("Pending Change     :", "YES" if pending_change else "NO")

    print(f"\nNDX Close           : {current_status['NDX Close']:.2f}")
    print(f"NDX MA30            : {current_status['NDX MA30']:.2f}")
    print("NDX > MA30          :", current_status["NDX > MA30"])
    print("NDX Cross Up        :", current_status["NDX Cross Up"])
    print("NDX Cross Down      :", current_status["NDX Cross Down"])

    print(f"\nSPX Month Close     : {current_status['SPX Month Close']:.2f}")
    print(f"SPX 12M MA          : {current_status['SPX 12M MA']:.2f}")
    print("SPX >= 12M MA       :", current_status["SPX >= 12M MA"])

    print("\nUnemployment Period :", current_status["Unemployment Reference Month"])
    print("Unemployment Release:", current_status["Unemployment Release Date"])
    print(f"Unemployment Rate   : {current_status['Unemployment Rate']:.2f}%")
    print(f"Unemployment 12M MA : {current_status['Unemployment 12M MA']:.2f}%")
    print("Unemp <= 12M MA     :", current_status["Unemployment <= 12M MA"])

    print("Macro Condition     :", current_status["Macro Condition"])
    print("Macro Cross Up      :", current_status["Macro Cross Up"])
    print("Macro Cross Down    :", current_status["Macro Cross Down"])

    print("SPY Condition       :", current_status["SPY Condition"])
    print("SPY Cross Up        :", current_status["SPY Cross Up"])
    print("SPY Cross Down      :", current_status["SPY Cross Down"])
    print("Reason              :", current_status["Reason"])

    print("\n" + "=" * 70)
    print("STATE MACHINE VALIDATION")
    print("=" * 70)
    print("Direct QQQ -> SPY switches :", validation["QQQ -> SPY"])
    print("Direct SPY -> QQQ switches :", validation["SPY -> QQQ"])
    print("✅ Every recorded Strategy 2 transition passed event validation")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, metrics in summary.items():
        print(f"\n{name}")
        print("  YTD Return    :", percent(metrics["YTD Return"]))
        print("  Since 2016    :", percent(metrics["Since 2016"]))
        print("  CAGR          :", percent(metrics["CAGR"]))
        print("  Max Drawdown  :", percent(metrics["Max Drawdown"]))

    print("\n" + "=" * 70)
    print("LATEST 12 MONTHS")
    print("=" * 70)
    print(monthly.tail(12).to_string(float_format=lambda x: f"{x * 100:.2f}%"))

    print("\n" + "=" * 70)
    print("ANNUAL RETURNS")
    print("=" * 70)
    print(annual.to_string(float_format=lambda x: f"{x * 100:.2f}%"))

    print(f"\nPosition changes: {len(position_history)}")
    print("\n✅ Strategy 2 event-driven backtest completed")


if __name__ == "__main__":
    main()
