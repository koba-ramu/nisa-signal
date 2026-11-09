"""株価・財務データを取得して data/market.json に保存する(段階2)。

GitHub Actions から毎朝7時30分(日本時間)に実行する。手元で試すときは:
    python scripts/fetch_data.py
"""
import datetime as dt
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "scripts" / "watchlist.json").read_text(encoding="utf-8"))
OUT = ROOT / "data" / "market.json"
JST = dt.timezone(dt.timedelta(hours=9))
MUFG_CSV = "https://www.am.mufg.jp/fund_file/setteirai/{code}.csv"


def num(x, n=2):
    """数値を丸める。欠けている値は None にする。"""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, n)


def pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def price_metrics(close, high=None, volume=None, fx=1.0):
    """株価の列から、値動き・騰落率・移動平均などを計算する。"""
    close = close.dropna()
    if len(close) < 30:
        return {}
    last = close.iloc[-1]

    def ago(n):
        return close.iloc[-1 - n] if len(close) > n else None

    def ma(n):
        return close.tail(n).mean() if len(close) >= n else None

    lr = np.log(close / close.shift(1)).dropna().tail(252)
    m = {
        "price": num(last, 4),
        "prev": num(close.iloc[-2], 4),
        "chg": num(pct(last, close.iloc[-2])),
        "date": close.index[-1].strftime("%Y-%m-%d"),
        "vol": num(lr.std() * math.sqrt(252) * 100, 1),   # 過去1年の値動き(年率%)
        "r1m": num(pct(last, ago(21)), 1),
        "r3m": num(pct(last, ago(63)), 1),
        "r1y": num(pct(last, ago(252)), 1),
        "ma5": num(ma(5), 4),
        "ma25": num(ma(25), 4),
        "ma75": num(ma(75), 4),
        "ma200v": num(ma(200), 4),
        "rsi14": num(rsi(close).iloc[-1], 1),
    }
    m["ma200"] = num(pct(last, m["ma200v"]), 1)   # 200日線からの乖離(%)
    hi = (high if high is not None else close).dropna().tail(252).max()
    m["dd"] = num(pct(last, hi), 1)                 # 1年の高値からの下落(%)
    if volume is not None:
        v = volume.dropna()
        if len(v) >= 65 and v.iloc[-65:-5].mean() > 0:
            m["volx"] = num(v.tail(5).mean() / v.iloc[-65:-5].mean(), 2)   # 出来高の倍率
        turnover = (close * volume).dropna().tail(20).mean()
        m["liq"] = num(turnover * fx / 1e8, 0)                             # 1日の売買代金(億円)
    m["closes"] = [num(x, 4) for x in close.tail(260).tolist()]
    return m


def fundamentals(tk, price):
    """PER・PBR・ROE・配当・自己資本比率・増益率・決算日を取得する。取れないものは None。"""
    f = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    f["per"] = num(info.get("trailingPE"), 1)
    f["pbr"] = num(info.get("priceToBook"), 2)
    # 異常値は捨てる(米国上場の外国株は、株価と純資産の通貨・株数の単位がずれてPBRが壊れることがある)
    if f["per"] is not None and not (0 < f["per"] < 300):
        f["per"] = None
    if f["pbr"] is not None and not (0 < f["pbr"] < 50):
        f["pbr"] = None
    roe = info.get("returnOnEquity")
    f["roe"] = num(roe * 100, 1) if roe is not None else None
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    if rate and price:
        f["div"] = num(rate / price * 100, 2)
    else:
        f["div"] = 0.0 if info else None
    eg = info.get("earningsGrowth")
    f["epsg"] = num(eg * 100, 1) if eg is not None else None   # 直近の増益率(前年同期比)
    try:
        bs = tk.balance_sheet
        eq = bs.loc["Stockholders Equity"].dropna().iloc[0]
        ta = bs.loc["Total Assets"].dropna().iloc[0]
        f["eq"] = num(eq / ta * 100, 1)                         # 自己資本比率(%)
    except Exception:
        f["eq"] = None
    f["earn"] = None
    try:
        cal = tk.calendar or {}
        today = dt.datetime.now(JST).date()
        future = [d for d in cal.get("Earnings Date", []) if d >= today]
        if future:
            f["earn"] = (min(future) - today).days               # 決算発表までの日数
    except Exception:
        pass
    return f


def fund_metrics(cfg):
    """三菱UFJアセットマネジメントのCSVから、投資信託の基準価額を取得する。"""
    res = requests.get(MUFG_CSV.format(code=cfg["mufg"]), timeout=30)
    res.raise_for_status()
    df = pd.read_csv(io.StringIO(res.content.decode("cp932")), skiprows=1)
    df.columns = ["date", "nav", "nav_re", "dist", "aum"]
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df = df.set_index("date")
    m = price_metrics(df["nav_re"].astype(float))
    m["price"] = num(df["nav"].iloc[-1], 0)       # 基準価額(1万口あたり)
    m["aum"] = num(df["aum"].iloc[-1], 0)         # 純資産総額(億円)
    m["fee"] = cfg["fee"]
    m["spread"] = cfg["spread"]
    return m


def main():
    out = {
        "updated": dt.datetime.now(JST).isoformat(timespec="minutes"),
        "source": "yfinance(Yahoo Finance)/ 三菱UFJアセットマネジメント",
        "usdjpy": None, "indices": {}, "items": {}, "errors": [],
    }
    tickers = [x["ticker"] for x in CFG["indices"]] + [x["ticker"] for x in CFG["stocks"]]
    hist = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                       progress=False, group_by="column", threads=True)
    close, high, volume = hist["Close"], hist["High"], hist["Volume"]

    fx = close["JPY=X"].dropna()
    usdjpy = float(fx.iloc[-1]) if len(fx) else None
    out["usdjpy"] = num(usdjpy, 2)

    for x in CFG["indices"]:
        c = close[x["ticker"]].dropna()
        if len(c) < 2:
            out["errors"].append(f"{x['name']}: データなし")
            continue
        out["indices"][x["id"]] = {
            "name": x["name"], "price": num(c.iloc[-1], 2),
            "chg": num(pct(c.iloc[-1], c.iloc[-2]), 2), "date": c.index[-1].strftime("%Y-%m-%d"),
        }

    for x in CFG["stocks"]:
        t = x["ticker"]
        try:
            rate = usdjpy if x["market"] == "us" else 1.0
            if rate is None:
                raise ValueError("ドル円が取れないため円換算できません")
            m = price_metrics(close[t], high[t], volume[t], rate)
            if not m:
                raise ValueError("株価データが足りません")
            m.update(fundamentals(yf.Ticker(t), m["price"]))
            out["items"][x["id"]] = m
        except Exception as e:
            out["errors"].append(f"{x['id']}: {e}")
        time.sleep(0.5)

    for x in CFG["funds"]:
        try:
            out["items"][x["id"]] = fund_metrics(x)
        except Exception as e:
            out["errors"].append(f"{x['id']}: {e}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    total = len(CFG["stocks"]) + len(CFG["funds"])
    print(f"取得: {len(out['items'])}/{total} 銘柄、指数 {len(out['indices'])} 件、ドル円 {out['usdjpy']}")
    for e in out["errors"]:
        print("  エラー:", e)
    if not out["items"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
