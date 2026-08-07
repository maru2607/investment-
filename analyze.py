"""
ETF買い時/売り時判定スクリプト

watchlist.json に登録された銘柄について、
- 52週高値からの下落率
- 200日移動平均線・乖離率
- RSI(14)
- CCI(20)
- MACD(ゴールデンクロス/デッドクロス)
- PER(取得できる場合のみ)
を計算し、シグナルをスコアリングして signals.json に出力する。

GitHub Actions から定期実行されることを想定。
"""

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST_PATH = ROOT / "watchlist.json"
SIGNALS_PATH = ROOT / "signals.json"

LOOKBACK_DAYS = 400  # 200日MAや52週高値計算に十分な余裕を持たせる


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def get_per(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        per = info.get("trailingPE") or info.get("forwardPE")
        return round(per, 2) if per else None
    except Exception:
        return None


def analyze_ticker(symbol: str, name: str):
    df = yf.download(symbol, period=f"{LOOKBACK_DAYS}d", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 210:
        return {
            "symbol": symbol,
            "name": name,
            "error": "十分な価格データを取得できませんでした",
        }

    # yfinance が MultiIndex 列を返す場合に対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ma200 = close.rolling(200).mean()
    high_52w = close.rolling(252, min_periods=100).max()
    drawdown_pct = (close.iloc[-1] - high_52w.iloc[-1]) / high_52w.iloc[-1] * 100

    rsi = calc_rsi(close).iloc[-1]
    cci = calc_cci(high, low, close).iloc[-1]
    macd, signal_line, hist = calc_macd(close)

    macd_cross = None
    if len(hist) > 2 and not hist.iloc[-2:].isna().any():
        prev_hist = hist.iloc[-2]
        curr_hist = hist.iloc[-1]
        if prev_hist < 0 <= curr_hist:
            macd_cross = "golden"  # デッドクロスからゴールデンクロスへ
        elif prev_hist > 0 >= curr_hist:
            macd_cross = "dead"

    ma200_last = ma200.iloc[-1]
    price_last = close.iloc[-1]
    ma200_dev_pct = (price_last - ma200_last) / ma200_last * 100 if pd.notna(ma200_last) else None

    per = get_per(symbol)

    # --- スコアリング ---
    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []

    if pd.notna(rsi):
        if rsi < 30:
            buy_score += 1
            reasons_buy.append(f"RSI({rsi:.1f})が30未満で売られすぎ")
        elif rsi > 70:
            sell_score += 1
            reasons_sell.append(f"RSI({rsi:.1f})が70超で買われすぎ")

    if pd.notna(cci):
        if cci < -100:
            buy_score += 1
            reasons_buy.append(f"CCI({cci:.1f})が-100未満で売られすぎ")
        elif cci > 100:
            sell_score += 1
            reasons_sell.append(f"CCI({cci:.1f})が+100超で買われすぎ")

    if pd.notna(drawdown_pct):
        if drawdown_pct <= -15:
            buy_score += 1
            reasons_buy.append(f"52週高値から{drawdown_pct:.1f}%下落(押し目)")
        if drawdown_pct <= -25:
            buy_score += 1
            reasons_buy.append("52週高値から25%超の大幅下落(暴落水準)")

    if ma200_dev_pct is not None:
        if ma200_dev_pct <= -10:
            buy_score += 1
            reasons_buy.append(f"200日線より{ma200_dev_pct:.1f}%下(トレンド割れ)")
        elif ma200_dev_pct >= 20:
            sell_score += 1
            reasons_sell.append(f"200日線より+{ma200_dev_pct:.1f}%上(過熱)")

    if macd_cross == "golden":
        buy_score += 1
        reasons_buy.append("MACDがゴールデンクロス")
    elif macd_cross == "dead":
        sell_score += 1
        reasons_sell.append("MACDがデッドクロス")

    if buy_score >= 2:
        signal = "buy"
    elif sell_score >= 2:
        signal = "sell"
    else:
        signal = "hold"

    return {
        "symbol": symbol,
        "name": name,
        "price": round(float(price_last), 2),
        "drawdown_from_52w_high_pct": round(float(drawdown_pct), 2) if pd.notna(drawdown_pct) else None,
        "ma200_deviation_pct": round(float(ma200_dev_pct), 2) if ma200_dev_pct is not None else None,
        "rsi14": round(float(rsi), 2) if pd.notna(rsi) else None,
        "cci20": round(float(cci), 2) if pd.notna(cci) else None,
        "macd_cross": macd_cross,
        "per": per,
        "signal": signal,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "reasons_buy": reasons_buy,
        "reasons_sell": reasons_sell,
    }


def main():
    watchlist = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    results = []
    for item in watchlist["tickers"]:
        try:
            result = analyze_ticker(item["symbol"], item.get("name", item["symbol"]))
        except Exception as e:
            result = {"symbol": item["symbol"], "name": item.get("name"), "error": str(e)}
        results.append(result)

    output = {
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "results": results,
        "has_active_signal": any(r.get("signal") in ("buy", "sell") for r in results),
    }

    SIGNALS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {SIGNALS_PATH} with {len(results)} tickers.")


if __name__ == "__main__":
    main()
