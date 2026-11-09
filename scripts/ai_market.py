"""Gemini で市況まとめ・銘柄ごとのAI市況判定・話題の銘柄探しを行い、data/market.json に書き足す(段階4)。

fetch_data.py のあとに実行する。必要な設定(GitHub の Secrets、手元なら 06_成果物/.env):
    GEMINI_API_KEY  … Gemini の API キー(必須)
    NTFY_TOPIC      … スマホ通知(ntfy)の宛先名(任意。なければ通知しない)
    APP_URL         … 通知を押したときに開くアプリのURL(任意)
    GEMINI_MODEL    … 使うモデルを固定したいとき(任意)

送るのは銘柄名・株価の動き・指数だけ。保有株数や資金額などの個人情報は送らない。
"""
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "market.json"
LOG = ROOT / "data" / "ai_log.csv"
CFG = json.loads((ROOT / "scripts" / "watchlist.json").read_text(encoding="utf-8"))
JST = dt.timezone(dt.timedelta(hours=9))
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BATCH = 4            # 1回の問い合わせで判定する銘柄数(少ないほど出典が混ざりにくい)
WAIT = 7             # 問い合わせの間隔(秒)。無料枠の「1分あたりの回数」を超えないため
BUDGET = 20 * 60    # AI部分の持ち時間(秒)。超えたら残りは判定せず、ここまでの結果を保存する
MAX_FAILS = 3        # 連続でこの回数失敗したら、今日はあきらめる(無料枠切れなど)
T0 = time.time()


def log(*a):
    """GitHub の画面ですぐ見えるよう、途中経過をその場で表示する。"""
    print(f"[{int(time.time() - T0):4d}秒]", *a, flush=True)


class QuotaError(RuntimeError):
    """1日の無料枠を使い切ったなど、今日はもうやり直しても無駄なとき。"""


def load_env():
    """手元で試すとき用: 06_成果物/.env から設定を読む(GitHub では Secrets を使う)。"""
    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()
KEY = os.environ.get("GEMINI_API_KEY", "").strip()
# 2026-11-09 の試運転で、無料枠では gemini-2.5-flash だけが Google検索付きで使えた(3.x系は 429)。
# そのため 2.5-flash を最初に試し、使えなくなったときだけ新しいモデルを試す。
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-3.6-flash",
                      "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"] if m]
_model = {"name": None}

SYSTEM = (
    "あなたは日本の個人投資家向けの市況アナリストです。"
    "必ずGoogle検索で直近7日以内のニュースを確かめ、事実にもとづいて答えてください。"
    "推測で事実を作ってはいけません。材料が見つからない、または判断できないときは0点・確信度「低」にしてください。"
    "あなたの役割はニュースや市況という材料の評価で、売買の指示はしません。"
    "出力は指定されたJSONだけにし、前後に説明文を付けないでください。日本語で答えてください。"
)


def gemini(prompt):
    """Google検索つきで Gemini に問い合わせ、(本文, 出典リスト) を返す。使えるモデルを順に試す。"""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2},
    }
    models = [_model["name"]] if _model["name"] else MODELS
    last = None
    for model in models:
        for attempt in range(2):
            t = time.time()
            try:
                r = requests.post(API.format(model=model), json=body, timeout=90,
                                  headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            except requests.RequestException as e:
                last = f"{model}: 通信エラー {e}"
                log("  ", last)
                time.sleep(10)
                continue
            log(f"   {model} → {r.status_code}({time.time() - t:.0f}秒)")
            if r.status_code == 429:                                # 回数制限
                msg = quota_message(r)
                last = f"{model}: 429 {msg}"
                log("     理由:", msg)
                # 「上限0」(無料枠でこのモデルが使えない)や1日の枠切れは、待っても無駄 → 次のモデルへ
                if re.search(r"limit: ?0\b|per ?day|PerDay|daily", r.text, re.I) or attempt == 1:
                    break
                time.sleep(30)
                continue
            if r.status_code >= 500:                                # 混雑 → 待ってやり直す
                last = f"{model}: {r.status_code}"
                time.sleep(15)
                continue
            if r.status_code in (400, 403, 404):                   # モデルが使えない・キーの問題
                last = f"{model}: {r.status_code} {r.text[:200]}"
                if r.status_code == 403 or "API key" in r.text:
                    raise RuntimeError("APIキーが無効か、権限がありません: " + r.text[:200])
                break
            r.raise_for_status()
            j = r.json()
            cand = (j.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            chunks = cand.get("groundingMetadata", {}).get("groundingChunks", []) or []
            sources = []
            for c in chunks:
                w = c.get("web") or {}
                if w.get("uri") and len(sources) < 6:
                    sources.append({"title": (w.get("title") or "")[:80], "url": w["uri"]})
            if _model["name"] != model:
                log(f"   使うモデル: {model}")
            _model["name"] = model
            return text, sources
    if last and ": 429" in last:
        raise QuotaError("どのモデルも回数制限(429)で使えませんでした: " + last)
    raise RuntimeError("Gemini に問い合わせできませんでした: " + str(last))


def quota_message(r):
    """429 の返事から、制限の種類(1分/1日/上限0など)が分かる部分だけ取り出す。キーは含まれない。"""
    try:
        err = r.json().get("error", {})
        ids = []
        for d in err.get("details", []):
            for v in d.get("violations", []) or []:
                ids.append(f"{v.get('quotaId', '')} 上限={v.get('quotaValue', '?')}")
        head = "制限の種類: " + "; ".join(ids) + " / " if ids else ""
        return (head + err.get("message", "")[:120]).replace("\n", " ")
    except ValueError:
        return r.text[:200].replace("\n", " ")


def diagnose():
    """全滅したときの原因切り分け: Google検索なしの短い問い合わせが通るか試す(1回だけ)。"""
    body = {"contents": [{"role": "user", "parts": [{"text": "「はい」とだけ答えてください。"}]}]}
    for model in MODELS[:2]:
        try:
            r = requests.post(API.format(model=model), json=body, timeout=60,
                              headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            log(f"診断: {model} を検索なしで → {r.status_code}"
                + ("(検索なしなら使える → 検索機能の枠の問題)" if r.ok else "(検索なしでも不可 → キー/プロジェクトの枠の問題)"))
            if not r.ok and r.status_code == 429:
                log("     理由:", quota_message(r))
        except requests.RequestException as e:
            log("診断: 通信エラー", e)
        time.sleep(3)


def parse_json(text):
    """返事の中から JSON を取り出す(```json ... ``` や前後の文章が付いていても読む)。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def summary(market):
    idx = market.get("indices", {})
    lines = [f"- {v['name']}: {v['price']}(前日比 {v['chg']:+.2f}%、{v['date']})" for v in idx.values() if v.get("chg") is not None]
    big = [v["name"] for k, v in idx.items() if k != "USDJPY" and v.get("chg") is not None and abs(v["chg"]) >= 3]
    prompt = (
        "今朝の市況を、日本の個人投資家向けに3行でまとめてください。対象は日本株・米国株・半導体株・為替です。\n"
        "直近の終値は次のとおりです(この数字は正しいものとして使ってください):\n" + "\n".join(lines) + "\n"
        + (f"大きく動いた指数があります({'、'.join(big)})。1行目でその理由を説明してください。\n" if big else "")
        + "あわせて、今日〜今週に注意すべき予定やニュース(決算・経済指標・政策など)があれば alerts に入れてください。\n"
        '出力形式: {"lines": ["1行目(60字以内)", "2行目", "3行目"], '
        '"alerts": [{"text": "内容(50字以内)", "level": "高 または 中"}]}'
    )
    data, sources = None, []
    text, sources = gemini(prompt)
    data = parse_json(text)
    out = {
        "lines": [str(x)[:90] for x in (data.get("lines") or [])][:3],
        "alerts": [{"text": str(a.get("text", ""))[:80], "level": "高" if a.get("level") == "高" else "中"}
                   for a in (data.get("alerts") or []) if a.get("text")][:5],
        "sources": sources, "big": big,
    }
    return out


def judge_batch(items):
    rows = []
    for x in items:
        mk = "日本株" if x["market"] == "jp" else "米国株"
        rows.append(f"- id={x['id']} {x['name']}({mk}、{x.get('code')})前日比 {x.get('chg')}%、1か月 {x.get('r1m')}%")
    prompt = (
        "次の銘柄それぞれについて、直近7日以内のニュース・決算・業界動向をGoogle検索で調べ、株価への影響を評価してください。\n"
        + "\n".join(rows) + "\n\n"
        "各銘柄について次を答えてください:\n"
        "- short: 数日〜数週間の株価への影響(-2〜2の整数。良い材料ほど大きい)\n"
        "- long: 数か月〜数年の株価への影響(-2〜2の整数)\n"
        "- conf: 判断の確信度(高・中・低)。材料が少ない・古いときは低\n"
        "- buzz: ニュースの話題度(1=良い話題が多い、0=目立たない、-1=悪い話題が多い)\n"
        "- severe: 不祥事、粉飾、業績予想の大幅な下方修正、上場廃止のおそれ、重大な訴訟・規制のどれかがあれば true\n"
        "- alert: 投資家がすぐ知るべき度合い(高・中・低)。決算の大きな驚き、severe に当たるもの、大型買収などは高\n"
        "- reason: 根拠(60字以内。具体的な出来事を書く)\n"
        "- headline: 最も重要なニュースの見出し(なければ空文字)\n"
        '出力形式: {"items": [{"id": "...", "short": 0, "long": 0, "conf": "中", "buzz": 0, '
        '"severe": false, "alert": "低", "reason": "...", "headline": "..."}]}'
    )
    text, sources = gemini(prompt)
    data = parse_json(text)
    out = {}
    ids = {x["id"] for x in items}
    for it in data.get("items") or []:
        i = str(it.get("id", ""))
        if i not in ids:
            continue
        out[i] = {
            "short": clamp_int(it.get("short"), -2, 2),
            "long": clamp_int(it.get("long"), -2, 2),
            "conf": it.get("conf") if it.get("conf") in ("高", "中", "低") else "低",
            "buzz": clamp_int(it.get("buzz"), -1, 1),
            "severe": bool(it.get("severe")),
            "alert": it.get("alert") if it.get("alert") in ("高", "中", "低") else "低",
            "reason": str(it.get("reason", ""))[:90],
            "headline": str(it.get("headline", ""))[:90],
            "sources": sources,
        }
    return out


def hot_stocks(known):
    prompt = (
        "いま日本と米国の株式市場で、半導体・AI関連を中心に話題になっている銘柄を5つ挙げてください。"
        "直近7日以内のニュースをGoogle検索で確かめてください。次の銘柄は除いてください: " + "、".join(sorted(known)) + "\n"
        '出力形式: {"hot": [{"name": "銘柄名", "ticker": "日本株は4桁コード、米国株はティッカー", '
        '"market": "jp または us", "reason": "話題の理由(50字以内)"}]}'
    )
    text, sources = gemini(prompt)
    data = parse_json(text)
    hot = []
    for h in (data.get("hot") or [])[:5]:
        if h.get("name") and h.get("ticker"):
            hot.append({"name": str(h["name"])[:40], "ticker": str(h["ticker"])[:12],
                        "market": "jp" if h.get("market") == "jp" else "us", "reason": str(h.get("reason", ""))[:80]})
    return {"date": dt.datetime.now(JST).date().isoformat(), "items": hot, "sources": sources}


def append_log(market, today):
    """AIの成績表のための記録。あとで「AIの判定のあと株価がどう動いたか」を確かめる。"""
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "id", "price", "ai_short", "ai_long", "conf", "tech_short", "tech_mid", "tech_long"])
        for i, m in market["items"].items():
            a, t = m.get("ai"), m.get("tech") or {}
            if not a:
                continue
            w.writerow([today, i, m.get("price"), a["short"], a["long"], a["conf"],
                        (t.get("short") or {}).get("score"), (t.get("mid") or {}).get("score"), (t.get("long") or {}).get("score")])


def notify(market):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("通知: NTFY_TOPIC が未設定のため送りません")
        return
    ai = market.get("ai", {})
    s = ai.get("summary") or {}
    today = dt.datetime.now(JST)
    body = list(s.get("lines", []))
    highs = [a["text"] for a in s.get("alerts", []) if a.get("level") == "高"]
    stock_alerts = [f"{m['name']}: {m['ai']['headline'] or m['ai']['reason']}"
                    for m in market["items"].values() if m.get("ai") and (m["ai"]["alert"] == "高" or m["ai"]["severe"])]
    if highs or stock_alerts:
        body.append("")
        body.append("【注目】")
        body += [f"・{t}" for t in (highs + stock_alerts)[:4]]
    msg = {"topic": topic, "title": f"今朝の市況まとめ({today.month}/{today.day})",
           "message": "\n".join(body) or "市況まとめを作れませんでした。アプリで確認してください。",
           "priority": 4 if s.get("big") or stock_alerts else 3, "tags": ["chart_with_upwards_trend"]}
    if os.environ.get("APP_URL"):
        msg["click"] = os.environ["APP_URL"]
    try:
        r = requests.post("https://ntfy.sh/", json=msg, timeout=30)
        print("通知:", r.status_code)
    except requests.RequestException as e:
        print("通知に失敗:", e)


def main():
    if not KEY:
        print("GEMINI_API_KEY が未設定のため、AIの市況判定はしません(株価データはそのまま使えます)")
        return
    market = json.loads(DATA.read_text(encoding="utf-8"))
    today = dt.datetime.now(JST).date()
    ai = market.get("ai") or {}
    errors = []

    stop = None          # 途中でやめた理由
    log("市況まとめ: 開始")
    try:
        ai["summary"] = summary(market)
        ai["summary"]["date"] = today.isoformat()
        log("市況まとめ: 完了")
    except QuotaError as e:
        errors.append(f"市況まとめ: {e}")
        stop = str(e)
    except Exception as e:
        errors.append(f"市況まとめ: {e}")
        log("市況まとめ: 失敗", e)

    stocks = [dict(m, id=i) for i, m in market["items"].items() if m.get("market") in ("jp", "us")]
    done, fails = 0, 0
    for n in range(0, len(stocks), BATCH):
        if stop:
            break
        if time.time() - T0 > BUDGET:
            stop = f"持ち時間({BUDGET // 60}分)を超えたため、{n}銘柄目以降は判定しませんでした"
            break
        time.sleep(WAIT)
        part = stocks[n:n + BATCH]
        log(f"銘柄判定 {n + 1}〜{n + len(part)}/{len(stocks)}")
        try:
            res = judge_batch(part)
            for i, a in res.items():
                a["date"] = today.isoformat()
                market["items"][i]["ai"] = a
                done += 1
            fails = 0
        except QuotaError as e:
            stop = str(e)
        except Exception as e:
            fails += 1
            errors.append(f"銘柄判定 {','.join(x['id'] for x in part)}: {e}")
            log("  失敗:", e)
            if fails >= MAX_FAILS:
                stop = f"{MAX_FAILS}回続けて失敗したため中止しました"
    if stop:
        errors.append(stop)
        log("中止:", stop)
        if done == 0:
            diagnose()

    hot = ai.get("hot")
    if not stop and (not hot or (today - dt.date.fromisoformat(hot["date"])).days >= 7):  # 話題の銘柄は週1回
        try:
            time.sleep(WAIT)
            log("話題の銘柄: 開始")
            ai["hot"] = hot_stocks({x["name"] for x in CFG["stocks"]})
        except Exception as e:
            errors.append(f"話題の銘柄: {e}")

    ai["model"] = _model["name"]
    ai["updated"] = dt.datetime.now(JST).isoformat(timespec="minutes")
    ai["errors"] = errors
    market["ai"] = ai
    DATA.write_text(json.dumps(market, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    append_log(market, today.isoformat())
    print(f"AI: モデル {_model['name']}、銘柄判定 {done}/{len(stocks)}、市況まとめ {'あり' if ai.get('summary') else 'なし'}")
    for e in errors:
        print("  エラー:", e)
    notify(market)
    if done == 0 and not ai.get("summary"):
        sys.exit(1)


if __name__ == "__main__":
    main()
