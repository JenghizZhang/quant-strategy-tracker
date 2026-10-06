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
#   SPY Condition = A OR B
#
#   入场：SPY Condition False -> True
#         且当天 NDX <= MA30（条件1不满足）
#   出场：SPY Condition True -> False
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
    ndx_state = pd.DataFrame({"NDX Close": ndx["Close"]})
    ndx_state["NDX MA30"] = ndx_state["NDX Close"].rolling(30, min_periods=30).mean()

    spx_monthly = build_spx_monthly_state(spx["Close"])

    macro_columns = [
        "Reference Month",
        "Release Date",
        "Unemployment Rate",
        "Unemployment 12M MA",
        "Unemployment <= 12M MA",
        "Release Date Source",
    ]
    missing = [c for c in macro_columns if c not in unemployment_daily.columns]
    if missing:
        raise RuntimeError("Missing unemployment columns: " + ", ".join(missing))

    signals = pd.concat(
        [ndx_state, spx_monthly, unemployment_daily[macro_columns]],
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
        raise RuntimeError("No common valid Strategy 2 signal dates.")

    # 当前状态
    signals["NDX > MA30"] = signals["NDX Close"] > signals["NDX MA30"]
    signals["SPX >= 12M MA"] = signals["SPX Month Close"] >= signals["SPX 12M MA"]
    signals["Unemployment <= 12M MA"] = (
        signals["Unemployment Rate"] <= signals["Unemployment 12M MA"]
    )
    signals["SPY Condition"] = (
        signals["SPX >= 12M MA"] | signals["Unemployment <= 12M MA"]
    )

    # 新鲜触发：第一条有效数据不算 cross
    prev_ndx_above = signals["NDX > MA30"].shift(1)
    prev_spy_condition = signals["SPY Condition"].shift(1)

    signals["NDX Cross Up"] = signals["NDX > MA30"] & prev_ndx_above.eq(False)
    signals["NDX Cross Down"] = (~signals["NDX > MA30"]) & prev_ndx_above.eq(True)
    signals["SPY Cross Up"] = signals["SPY Condition"] & prev_spy_condition.eq(False)
    signals["SPY Cross Down"] = (~signals["SPY Condition"]) & prev_spy_condition.eq(True)

    return signals


# ============================================================
# 状态机
# ============================================================

def apply_state_machine(signals, initial_holding=INITIAL_HOLDING):
    result = signals.copy()
    current_holding = initial_holding

    current_holdings = []
    next_holdings = []
    reasons = []

    for _, row in result.iterrows():
        holding = current_holding
        ndx_above = bool(row["NDX > MA30"])
        ndx_up = bool(row["NDX Cross Up"])
        ndx_down = bool(row["NDX Cross Down"])
        spy_condition = bool(row["SPY Condition"])
        spy_up = bool(row["SPY Cross Up"])
        spy_down = bool(row["SPY Cross Down"])

        next_holding = holding

        if holding == "QQQ":
            if ndx_down:
                if spy_up:
                    next_holding = "SPY"
                    reason = (
                        "NDX crossed to/below MA30 and SPY Condition crossed "
                        "False→True on the same day → QQQ → SPY"
                    )
                else:
                    next_holding = DEFENSIVE_ASSET
                    reason = f"NDX crossed to/below MA30 → QQQ exit → {DEFENSIVE_ASSET}"
            else:
                next_holding = "QQQ"
                reason = "Hold QQQ — no NDX down-cross to/below MA30"

        elif holding == "SPY":
            if spy_down:
                if ndx_up:
                    next_holding = "QQQ"
                    reason = (
                        "SPY Condition crossed True→False and NDX crossed above MA30 "
                        "on the same day → SPY → QQQ"
                    )
                else:
                    next_holding = DEFENSIVE_ASSET
                    reason = f"SPY Condition crossed True→False → SPY exit → {DEFENSIVE_ASSET}"
            else:
                next_holding = "SPY"
                reason = "Hold SPY — no SPY Condition True→False exit"

        elif holding == INITIAL_HOLDING:
            # CASH 只作为最早 seed。条件1优先。
            if ndx_up:
                next_holding = "QQQ"
                reason = "NDX crossed above MA30 → initial QQQ entry"
            elif spy_up and not ndx_above:
                next_holding = "SPY"
                reason = "SPY Condition crossed False→True and NDX <= MA30 → initial SPY entry"
            else:
                next_holding = INITIAL_HOLDING
                if spy_up and ndx_above:
                    reason = (
                        "SPY Condition crossed False→True, but Rule 1 is satisfied "
                        "(NDX > MA30) → stay CASH"
                    )
                else:
                    reason = "No fresh valid initial entry event → stay CASH"

        elif holding == DEFENSIVE_ASSET:
            # AGG 等待新的触发，条件1优先。
            if ndx_up:
                next_holding = "QQQ"
                reason = "NDX crossed above MA30 → QQQ entry"
            elif spy_up and not ndx_above:
                next_holding = "SPY"
                reason = "SPY Condition crossed False→True and NDX <= MA30 → SPY entry"
            else:
                next_holding = DEFENSIVE_ASSET
                if spy_up and ndx_above:
                    reason = (
                        "SPY Condition crossed False→True, but Rule 1 is satisfied "
                        f"(NDX > MA30) → stay {DEFENSIVE_ASSET}"
                    )
                elif spy_condition:
                    reason = (
                        "SPY Condition is already True but did not freshly trigger today "
                        f"→ stay {DEFENSIVE_ASSET}"
                    )
                else:
                    reason = f"No fresh valid entry event → stay {DEFENSIVE_ASSET}"

        else:
            raise RuntimeError(f"Unknown holding state: {holding}")

        current_holdings.append(holding)
        next_holdings.append(next_holding)
        reasons.append(reason)
        current_holding = next_holding

    result["Current Holding"] = current_holdings
    result["Signal"] = next_holdings
    result["Reason"] = reasons
    return result


# ============================================================
# 状态机硬校验
# ============================================================

def validate_state_machine(decisions, position_history):
    qqq_to_spy = 0
    spy_to_qqq = 0

    for _, change in position_history.iterrows():
        old_holding = change["From"]
        new_holding = change["To"]

        # 2016回测展示初始化，不当作历史真实换仓来校验。
        if old_holding == INITIAL_HOLDING:
            continue

        signal_date = pd.Timestamp(change["Signal Date"])
        if signal_date not in decisions.index:
            raise RuntimeError(
                f"Position History signal date missing from decisions: {signal_date.date()}"
            )

        row = decisions.loc[signal_date]
        ndx_above = bool(row["NDX > MA30"])
        ndx_up = bool(row["NDX Cross Up"])
        ndx_down = bool(row["NDX Cross Down"])
        spy_up = bool(row["SPY Cross Up"])
        spy_down = bool(row["SPY Cross Down"])

        if old_holding == "QQQ":
            if new_holding == "SPY":
                qqq_to_spy += 1
                if not (ndx_down and spy_up):
                    raise RuntimeError(
                        f"Illegal QQQ -> SPY on {signal_date.date()}: "
                        "requires same-day QQQ exit + fresh SPY entry."
                    )
            elif new_holding == DEFENSIVE_ASSET:
                if not ndx_down:
                    raise RuntimeError(
                        f"Illegal QQQ -> AGG on {signal_date.date()}: no NDX down-cross."
                    )
            else:
                raise RuntimeError(f"Illegal QQQ transition: {old_holding} -> {new_holding}")

        elif old_holding == "SPY":
            if new_holding == "QQQ":
                spy_to_qqq += 1
                if not (spy_down and ndx_up):
                    raise RuntimeError(
                        f"Illegal SPY -> QQQ on {signal_date.date()}: "
                        "requires same-day SPY exit + fresh QQQ entry."
                    )
            elif new_holding == DEFENSIVE_ASSET:
                if not spy_down:
                    raise RuntimeError(
                        f"Illegal SPY -> AGG on {signal_date.date()}: no SPY down-cross."
                    )
            else:
                raise RuntimeError(f"Illegal SPY transition: {old_holding} -> {new_holding}")

        elif old_holding == DEFENSIVE_ASSET:
            if new_holding == "QQQ":
                if not ndx_up:
                    raise RuntimeError(
                        f"Illegal AGG -> QQQ on {signal_date.date()}: no fresh NDX up-cross."
                    )
            elif new_holding == "SPY":
                if not (spy_up and not ndx_above):
                    raise RuntimeError(
                        f"Illegal AGG -> SPY on {signal_date.date()}: requires fresh "
                        "SPY up-cross and NDX <= MA30."
                    )
            else:
                raise RuntimeError(f"Illegal AGG transition: {old_holding} -> {new_holding}")
        else:
            raise RuntimeError(f"Unexpected prior holding: {old_holding}")

    return {"QQQ -> SPY": qqq_to_spy, "SPY -> QQQ": spy_to_qqq}


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
        common_index = common_index.intersection(prices[ticker].index)
    common_index = common_index.sort_values()

    if len(common_index) < 2:
        raise RuntimeError("Not enough common trading dates to run Strategy 2.")

    # 先在完整 warm-up 历史上运行状态机，再从2016开始展示。
    decisions = apply_state_machine(signals.loc[common_index].copy())

    execution = pd.DataFrame(index=common_index)
    execution["Target"] = decisions["Signal"].shift(1)
    execution["Reason"] = decisions["Reason"].shift(1)
    execution["Signal Date"] = pd.Series(common_index, index=common_index).shift(1)

    execution = execution.loc[execution.index >= BACKTEST_START].copy()
    execution = execution.dropna(subset=["Target", "Signal Date"])
    if execution.empty:
        raise RuntimeError("No Strategy 2 backtest dates available after 2016.")

    expected_current = decisions.loc[execution.index, "Current Holding"]
    mismatch = execution["Target"] != expected_current
    if mismatch.any():
        bad_date = mismatch[mismatch].index[0]
        raise RuntimeError(f"State-machine/execution mismatch on {bad_date.date()}.")

    equity = 1.0
    holding = None
    previous_date = None
    daily_rows = []
    position_history = []

    for date, row in execution.iterrows():
        target = row["Target"]
        signal_date = pd.Timestamp(row["Signal Date"])
        reason = row["Reason"]

        if previous_date is None:
            holding = target
            equity = apply_intraday_return(equity, holding, prices, date)

            if holding != INITIAL_HOLDING:
                position_history.append(
                    {
                        "Change": f"{INITIAL_HOLDING} → {holding}",
                        "Signal Date": signal_date,
                        "Execute Date": date,
                        "From": INITIAL_HOLDING,
                        "To": holding,
                        "Reason": (
                            "2016 reporting initialization from pre-2016 warm-up state; "
                            + str(reason)
                        ),
                    }
                )
        else:
            # 昨收 -> 今开属于旧持仓。
            equity = apply_overnight_return(
                equity, holding, prices, previous_date, date
            )

            # 今开执行昨天收盘信号。
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

            # 今开 -> 今收属于新持仓。
            equity = apply_intraday_return(equity, holding, prices, date)

        d = decisions.loc[date]
        daily_rows.append(
            {
                "Date": date,
                "Equity": equity,
                "Holding": holding,
                "Target": target,
                "Signal Date": signal_date,
                "Signal Reason": reason,
                "Today's Signal": d["Signal"],
                "Today's Reason": d["Reason"],
                "NDX Close": d["NDX Close"],
                "NDX MA30": d["NDX MA30"],
                "NDX > MA30": d["NDX > MA30"],
                "NDX Cross Up": d["NDX Cross Up"],
                "NDX Cross Down": d["NDX Cross Down"],
                "SPX Month Close": d["SPX Month Close"],
                "SPX 12M MA": d["SPX 12M MA"],
                "SPX >= 12M MA": d["SPX >= 12M MA"],
                "Unemployment Reference Month": str(d["Reference Month"]),
                "Unemployment Release Date": pd.Timestamp(d["Release Date"]),
                "Unemployment Rate": d["Unemployment Rate"],
                "Unemployment 12M MA": d["Unemployment 12M MA"],
                "Unemployment <= 12M MA": d["Unemployment <= 12M MA"],
                "SPY Condition": d["SPY Condition"],
                "SPY Cross Up": d["SPY Cross Up"],
                "SPY Cross Down": d["SPY Cross Down"],
            }
        )
        previous_date = date

    daily = pd.DataFrame(daily_rows).set_index("Date")
    history = pd.DataFrame(
        position_history,
        columns=["Change", "Signal Date", "Execute Date", "From", "To", "Reason"],
    )
    validation = validate_state_machine(decisions, history)
    return daily, history, decisions, validation


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

    print("\nSPY Condition       :", current_status["SPY Condition"])
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
