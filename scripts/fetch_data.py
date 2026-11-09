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
TOUSHIN_CSV = "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download"


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


def sma(s, n):
    return s.rolling(n).mean()


def crossed(a, b, days):
    """a が b を直近 days 日以内に上抜けしたら (1, 何日前)、下抜けしたら (-1, 何日前)、なければ (0, None)。"""
    d = (a - b).dropna().tail(days + 1)
    sign = np.sign(d.values)
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] > 0 and sign[i - 1] <= 0:
            return 1, len(sign) - 1 - i
        if sign[i] < 0 and sign[i - 1] >= 0:
            return -1, len(sign) - 1 - i
    return 0, None


def clamp_score(x):
    """点数を −2〜+2 の 0.5 刻みにそろえる。"""
    return max(-2.0, min(2.0, round(x * 2) / 2))


def ago_text(n):
    return "今日" if n == 0 else f"{n}日前"


def tech_scores(close, m, fund=False):
    """スタイル別のテクニカル判定(−2〜+2)と、その理由を返す。開発方針 4-0 のルール。"""
    if fund:
        dd = m.get("dd")
        s, why = 0.0, []
        if dd is not None:
            if dd <= -10:
                s = 1.5
                why.append(f"高値から{abs(dd)}%下落(追加購入の検討どき)")
            elif dd <= -5:
                s = 0.5
                why.append(f"高値から{abs(dd)}%下落(少し割安)")
            else:
                why.append(f"高値から{abs(dd)}%の下落にとどまる(大きな下落なし)")
        why.append("投資信託は原則として売らない(目標額に達したとき・資金が必要なときだけ)")
        t = {"score": clamp_score(s), "why": why}
        return {"short": t, "mid": t, "long": t}

    c = close.dropna()
    last = c.iloc[-1]
    ma5, ma25, ma75, ma200 = sma(c, 5), sma(c, 25), sma(c, 75), sma(c, 200)
    out = {}

    # 短期: RSI、5日線と25日線、ボリンジャーバンド
    s, why = 0.0, []
    r = rsi(c)
    rv, rp = r.iloc[-1], r.iloc[-2]
    if rv <= 30:
        if rv > rp:
            s += 1
            why.append(f"RSI {rv:.0f}:売られすぎから上向きに転じた")
        else:
            s += 0.5
            why.append(f"RSI {rv:.0f}:売られすぎ(まだ下向き)")
    elif rv >= 70:
        s -= 1
        why.append(f"RSI {rv:.0f}:買われすぎ")
    else:
        why.append(f"RSI {rv:.0f}:中立")
    x, n = crossed(ma5, ma25, 3)
    if x == 1:
        s += 1
        why.append(f"5日線が25日線を上抜けた({ago_text(n)})")
    elif x == -1:
        s -= 1
        why.append(f"5日線が25日線を下抜けた({ago_text(n)})")
    elif ma5.iloc[-1] > ma25.iloc[-1]:
        s += 0.5
        why.append("5日線が25日線より上")
    else:
        s -= 0.5
        why.append("5日線が25日線より下")
    mid, sd = sma(c, 20), c.rolling(20).std()
    if last <= (mid - 2 * sd).iloc[-1]:
        s += 0.5
        why.append("ボリンジャーバンドの下限(−2σ)に到達")
    elif last >= (mid + 2 * sd).iloc[-1]:
        s -= 0.5
        why.append("ボリンジャーバンドの上限(+2σ)に到達")
    out["short"] = {"score": clamp_score(s), "why": why}

    # 中期: 25日線と75日線、MACD
    s, why = 0.0, []
    x, n = crossed(ma25, ma75, 5)
    if x == 1:
        s += 1.5
        why.append(f"ゴールデンクロス:25日線が75日線を上抜けた({ago_text(n)})")
    elif x == -1:
        s -= 1.5
        why.append(f"デッドクロス:25日線が75日線を下抜けた({ago_text(n)})")
    elif ma25.iloc[-1] > ma75.iloc[-1]:
        s += 0.5
        why.append("25日線が75日線より上(上昇の流れ)")
    else:
        s -= 0.5
        why.append("25日線が75日線より下(下落の流れ)")
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    x, n = crossed(macd, signal, 3)
    if x == 1:
        s += 1
        why.append(f"MACDがシグナルを上抜けた({ago_text(n)})")
    elif x == -1:
        s -= 1
        why.append(f"MACDがシグナルを下抜けた({ago_text(n)})")
    elif macd.iloc[-1] > signal.iloc[-1]:
        s += 0.5
        why.append("MACDがシグナルより上")
    else:
        s -= 0.5
        why.append("MACDがシグナルより下")
    out["mid"] = {"score": clamp_score(s), "why": why}

    # 長期: 200日線との乖離と向き、業績の悪化
    s, why = 0.0, []
    if len(c) >= 221 and not math.isnan(ma200.iloc[-1]):
        dev = (last / ma200.iloc[-1] - 1) * 100
        slope = (ma200.iloc[-1] / ma200.iloc[-21] - 1) * 100
        if dev <= -20:
            s += 1
            why.append(f"200日線より{abs(dev):.0f}%下:大きな押し目")
        elif dev <= -10:
            s += 0.5
            why.append(f"200日線より{abs(dev):.0f}%下:押し目")
        elif dev >= 30:
            s -= 1
            why.append(f"200日線より{dev:.0f}%上:上がりすぎ")
        elif dev >= 15:
            s -= 0.5
            why.append(f"200日線より{dev:.0f}%上:やや上がりすぎ")
        else:
            why.append(f"200日線の近く({dev:+.0f}%)")
        if slope > 0:
            s += 0.5
            why.append("200日線は上向き(長期の上昇が続いている)")
        else:
            s -= 0.5
            why.append("200日線は下向き(長期の流れが弱い)")
    epsg = m.get("epsg")
    if epsg is not None and epsg <= -20:
        s -= 1
        why.append(f"業績が悪化している(増益率 {epsg:+.0f}%)")
    out["long"] = {"score": clamp_score(s), "why": why}
    return out


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
    f["divRate"] = num(rate, 4) if rate else (0.0 if info else None)   # 1株あたり年間配当(利回りは毎日の株価で計算)
    eg = info.get("earningsGrowth")
    f["epsg"] = num(eg * 100, 1) if eg is not None else None   # 直近の増益率(前年同期比)
    try:
        bs = tk.balance_sheet
        eq = bs.loc["Stockholders Equity"].dropna().iloc[0]
        ta = bs.loc["Total Assets"].dropna().iloc[0]
        f["eq"] = num(eq / ta * 100, 1)                         # 自己資本比率(%)
    except Exception:
        f["eq"] = None
    f["earnDate"] = None
    try:
        cal = tk.calendar or {}
        today = dt.datetime.now(JST).date()
        future = [d for d in cal.get("Earnings Date", []) if d >= today]
        if future:
            f["earnDate"] = min(future).isoformat()               # 次の決算発表日
    except Exception:
        pass
    f["fundDate"] = dt.datetime.now(JST).date().isoformat()     # 財務データを取った日
    if not info:
        raise RuntimeError("財務データが空(取得制限の可能性)")
    return f


FUND_KEYS = ["per", "pbr", "roe", "divRate", "epsg", "eq", "earnDate", "fundDate"]
ROTATE_DAYS = 5          # 財務データは5営業日で全銘柄を一巡する(1日あたり約20銘柄)
FUND_MAX_AGE = 8         # これより古い財務データは、順番に関係なく取り直す


def fundamentals_cached(tk, price, prev, index):
    """財務データは毎日取らず、順番に更新する。取れなければ前回の値を使う。"""
    today = dt.datetime.now(JST).date()
    old = {k: prev.get(k) for k in FUND_KEYS} if prev and prev.get("fundDate") else None
    age = (today - dt.date.fromisoformat(old["fundDate"])).days if old else None
    due = old is None or age >= FUND_MAX_AGE or index % ROTATE_DAYS == today.toordinal() % ROTATE_DAYS
    if not due:
        return old, False
    for wait in (0, 5, 15):           # 取得制限にかかったら、少し待ってやり直す
        if wait:
            time.sleep(wait)
        try:
            return fundamentals(tk, price), True
        except Exception:
            continue
    if old:
        return old, False
    return {k: None for k in FUND_KEYS}, False


def finish_fundamentals(m):
    """保存した財務データから、毎日変わる値(配当利回り・決算までの日数)を計算する。"""
    rate, price = m.get("divRate"), m.get("price")
    m["div"] = num(rate / price * 100, 2) if rate and price else (0.0 if rate == 0 else None)
    m["earn"] = None
    if m.get("earnDate"):
        days = (dt.date.fromisoformat(m["earnDate"]) - dt.datetime.now(JST).date()).days
        m["earn"] = days if days >= 0 else None


UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
      "Accept": "text/csv,*/*", "Accept-Language": "ja,en;q=0.8"}


def fund_table_toushin(cfg):
    """投資信託協会(投信総合検索ライブラリー)のCSV。GitHub のサーバーからも取れる。
    分配金再投資の価額はないが、eMAXIS Slim は分配金を出していないので基準価額と同じになる。"""
    res = requests.get(TOUSHIN_CSV, params={"isinCd": cfg["isin"], "associFundCd": cfg["assoc"]},
                       timeout=30, headers=UA)
    res.raise_for_status()
    df = pd.read_csv(io.StringIO(res.content.decode("cp932")))
    df = df.iloc[:, :3]
    df.columns = ["date", "nav", "aum"]
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")
    df["nav_re"] = df["nav"]
    df["aum"] = df["aum"] / 100                     # 百万円 → 億円
    return df


def fund_table_mufg(cfg):
    """三菱UFJアセットマネジメントのCSV(予備)。GitHub のサーバーからは 403 で断られることがある。"""
    res = requests.get(MUFG_CSV.format(code=cfg["mufg"]), timeout=30,
                       headers=dict(UA, Referer="https://www.am.mufg.jp/"))
    res.raise_for_status()
    df = pd.read_csv(io.StringIO(res.content.decode("cp932")), skiprows=1)
    df.columns = ["date", "nav", "nav_re", "dist", "aum"]
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    return df


def fund_metrics(cfg):
    """投資信託の基準価額を取得する(投資信託協会 → だめなら三菱UFJアセットマネジメント)。"""
    try:
        df = fund_table_toushin(cfg)
    except Exception as e:
        print(f"  {cfg['id']}: 投資信託協会から取れないため予備を使用 ({e})")
        df = fund_table_mufg(cfg)
    df = df.sort_values("date").set_index("date")
    m = price_metrics(df["nav_re"].astype(float))
    m["price"] = num(df["nav"].iloc[-1], 0)       # 基準価額(1万口あたり)
    m["aum"] = num(df["aum"].iloc[-1], 0)         # 純資産総額(億円)
    m["fee"] = cfg["fee"]
    m["spread"] = cfg["spread"]
    m["tech"] = tech_scores(None, m, fund=True)
    return m


def main():
    out = {
        "updated": dt.datetime.now(JST).isoformat(timespec="minutes"),
        "source": "yfinance(Yahoo Finance)/ 投資信託協会",
        "usdjpy": None, "indices": {}, "items": {}, "errors": [],
    }
    # 前回のデータ(財務データの使い回しと、取得に失敗したときの予備)
    prev, prev_ai = {}, None
    if OUT.exists():
        try:
            old_all = json.loads(OUT.read_text(encoding="utf-8"))
            prev, prev_ai = old_all.get("items", {}), old_all.get("ai")
        except Exception:
            prev = {}
    if prev_ai:
        out["ai"] = prev_ai          # AIの市況まとめ・話題の銘柄は ai_market.py が上書きするまで前回のものを残す

    tickers = [x["ticker"] for x in CFG["indices"]] + [x["ticker"] for x in CFG["stocks"]]
    hist = None
    for wait in (0, 30, 90):          # 株価は全銘柄まとめて1回で取る。失敗したら待ってやり直す
        if wait:
            time.sleep(wait)
        try:
            hist = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                               progress=False, group_by="column", threads=True)
            if hist is not None and not hist.empty:
                break
        except Exception as e:
            print("株価の取得に失敗、やり直します:", e)
    if hist is None or hist.empty:
        print("株価を取得できませんでした。前回のデータをそのまま残します。")
        sys.exit(1)
    close, high, volume = hist["Close"], hist["High"], hist["Volume"]
    fund_new = fund_kept = 0

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

    for idx, x in enumerate(CFG["stocks"]):
        t = x["ticker"]
        try:
            rate = usdjpy if x["market"] == "us" else 1.0
            if rate is None:
                raise ValueError("ドル円が取れないため円換算できません")
            m = price_metrics(close[t], high[t], volume[t], rate) if t in close else {}
            if not m:
                raise ValueError("株価データが足りません")
            f, fresh = fundamentals_cached(yf.Ticker(t), m["price"], prev.get(x["id"]), idx)
            m.update(f)
            finish_fundamentals(m)
            if fresh:
                fund_new += 1
                time.sleep(0.5)
            else:
                fund_kept += 1
            m["tech"] = tech_scores(close[t], m)
            m.update({"name": x["name"], "sector": x["sector"], "market": x["market"], "code": x["id"]})
            out["items"][x["id"]] = m
        except Exception as e:
            if x["id"] in prev:                       # 取れなかった銘柄は前回のデータを残す
                old = dict(prev[x["id"]])
                old["stale"] = True
                out["items"][x["id"]] = old
                out["errors"].append(f"{x['id']}: {e}(前回のデータを使用)")
            else:
                out["errors"].append(f"{x['id']}: {e}")

    for x in CFG["funds"]:
        try:
            out["items"][x["id"]] = fund_metrics(x)
        except Exception as e:
            if x["id"] in prev:
                old = dict(prev[x["id"]])
                old["stale"] = True
                out["items"][x["id"]] = old
                out["errors"].append(f"{x['id']}: {e}(前回のデータを使用)")
            else:
                out["errors"].append(f"{x['id']}: {e}")

    # 銘柄ごとのAI判定は毎日全銘柄はできない(無料枠)ため、前回の判定を日付つきで引き継ぐ
    for k, m in out["items"].items():
        if "ai" not in m and (prev.get(k) or {}).get("ai"):
            m["ai"] = prev[k]["ai"]

    total = len(CFG["stocks"]) + len(CFG["funds"])
    if not out["items"]:
        print("1銘柄も取得できませんでした。前回のデータをそのまま残します。")
        sys.exit(1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"取得: {len(out['items'])}/{total} 銘柄、指数 {len(out['indices'])} 件、ドル円 {out['usdjpy']}")
    print(f"財務データ: 今日更新 {fund_new} 銘柄、前回の値を使用 {fund_kept} 銘柄")
    for e in out["errors"]:
        print("  エラー:", e)


if __name__ == "__main__":
    main()
