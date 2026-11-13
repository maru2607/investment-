"""
ETF押し目判定スクリプト

watchlist.json に登録された銘柄について、平日22:30のシグナルチェックと
同じ5指標で「押し目かどうか」だけを判定し、signals.json に出力する。

判定指標(該当数を数える方式。売りシグナルは出さない):
- RSI(14) が 35 以下
- CCI(14) が -100 以下(ボラティリティの高い銘柄は -150 以下)
- 50日移動平均線からの乖離が -3% 以下
- 200日移動平均線からの乖離が -5% 以下
- 過去1年(252営業日)高値からの下落率が -10% 以下

該当 2つ以上 = 「強めの押し目」、1つ = 「押し目」、0 = 「押し目なし」。
これはスポット買いの目安であり、買い推奨ではない。

GitHub Actions から定期実行されることを想定。
"""

import json
import os
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# GitHub Actions ではリポジトリのチェックアウト先が GITHUB_WORKSPACE に入っているので、
# それを優先的に使う(analyze.py の設置場所が多少ズレても watchlist.json を正しく見つけられるように)。
_workspace = os.environ.get("GITHUB_WORKSPACE")
ROOT = Path(_workspace) if _workspace else Path(__file__).resolve().parent
WATCHLIST_PATH = ROOT / "watchlist.json"
SIGNALS_PATH = ROOT / "signals.json"

LOOKBACK_DAYS = 730  # 200日MAと252日高値を安定して計算するため2年分取得する

# yfinance のシンボル表記ゆれ対策。
# 「BTC」「ETH」のままだと暗号資産ではない別証券を指してしまうため、正式シンボルへ寄せる。
SYMBOL_ALIASES = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

# ボラティリティが高く、CCI -100 ではシグナルが頻出しすぎる銘柄。
# 過去3年の実測で、CCI(14) が -150 以下になる日の割合が全銘柄で 6〜8% に揃うため、
# この4銘柄だけ CCI の閾値を -150 にする(他は -100)。
HIGH_VOLATILITY_SYMBOLS = {"BTC-USD", "ETH-USD", "SPXL", "SOXL"}
LEVERAGED_SYMBOLS = {"SPXL", "SOXL"}

CCI_THRESHOLD_NORMAL = -100
CCI_THRESHOLD_HIGH_VOL = -150

RSI_THRESHOLD = 35
MA50_DEV_THRESHOLD = -3
MA200_DEV_THRESHOLD = -5
DRAWDOWN_THRESHOLD = -10


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def analyze_ticker(symbol: str, name: str):
    fetch_symbol = SYMBOL_ALIASES.get(symbol, symbol)
    high_vol = fetch_symbol in HIGH_VOLATILITY_SYMBOLS
    cci_threshold = CCI_THRESHOLD_HIGH_VOL if high_vol else CCI_THRESHOLD_NORMAL

    df = yf.download(fetch_symbol, period=f"{LOOKBACK_DAYS}d", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 260:
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

    price_last = float(close.iloc[-1])

    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    ma50_dev_pct = (price_last - ma50) / ma50 * 100 if pd.notna(ma50) else None
    ma200_dev_pct = (price_last - ma200) / ma200 * 100 if pd.notna(ma200) else None

    high_1y = close.rolling(252, min_periods=100).max().iloc[-1]
    drawdown_pct = (price_last - high_1y) / high_1y * 100 if pd.notna(high_1y) else None

    rsi = calc_rsi(close).iloc[-1]
    cci = calc_cci(high, low, close).iloc[-1]

    # --- 5指標の該当判定 ---
    hits = []

    if pd.notna(rsi) and rsi <= RSI_THRESHOLD:
        hits.append(f"RSI({rsi:.1f})が{RSI_THRESHOLD}以下で売られすぎ")

    if pd.notna(cci) and cci <= cci_threshold:
        hits.append(f"CCI({cci:.1f})が{cci_threshold}以下で売られすぎ")

    if ma50_dev_pct is not None and ma50_dev_pct <= MA50_DEV_THRESHOLD:
        hits.append(f"50日線より{ma50_dev_pct:.1f}%下")

    if ma200_dev_pct is not None and ma200_dev_pct <= MA200_DEV_THRESHOLD:
        hits.append(f"200日線より{ma200_dev_pct:.1f}%下")

    if drawdown_pct is not None and drawdown_pct <= DRAWDOWN_THRESHOLD:
        hits.append(f"1年高値から{drawdown_pct:.1f}%下落")

    hit_count = len(hits)
    if hit_count >= 2:
        signal = "strong"
    elif hit_count == 1:
        signal = "dip"
    else:
        signal = "none"

    # 長期トレンドの状態(200日線との位置関係)。押し目の質を見分けるための併記情報。
    if ma200_dev_pct is None:
        trend_label = None
    elif ma200_dev_pct <= MA200_DEV_THRESHOLD:
        trend_label = "長期トレンドが崩れた下落"
    else:
        trend_label = "長期トレンドを維持した押し目"

    return {
        "symbol": symbol,
        "name": name,
        "price": round(price_last, 2),
        "rsi14": round(float(rsi), 2) if pd.notna(rsi) else None,
        "cci14": round(float(cci), 2) if pd.notna(cci) else None,
        "cci_threshold": cci_threshold,
        "ma50_deviation_pct": round(float(ma50_dev_pct), 2) if ma50_dev_pct is not None else None,
        "ma200_deviation_pct": round(float(ma200_dev_pct), 2) if ma200_dev_pct is not None else None,
        "drawdown_from_1y_high_pct": round(float(drawdown_pct), 2) if drawdown_pct is not None else None,
        "high_volatility": high_vol,
        "leveraged": fetch_symbol in LEVERAGED_SYMBOLS,
        "trend_label": trend_label,
        "signal": signal,
        "hit_count": hit_count,
        "hits": hits,
    }


def _sanitize(obj):
    """NaN/Infinityなど非JSON準拠の値を再帰的にNoneへ置換する(壊れたJSON出力を防ぐ安全策)。"""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


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
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "results": results,
        "has_active_signal": any(r.get("signal") in ("dip", "strong") for r in results),
    }

    output = _sanitize(output)

    # allow_nan=False: 万一NaN/Infinityが紛れ込んだ場合、ここで例外を出してActionsのログに
    # 表示させる(黙って壊れたJSONを書き出さないようにするため)
    SIGNALS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {SIGNALS_PATH} with {len(results)} tickers.")


if __name__ == "__main__":
    main()
