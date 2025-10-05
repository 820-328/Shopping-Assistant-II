# pyright: reportMissingImports=false, reportMissingModuleSource=false
# Shopping Assistant + Auto Floor-Map Overlay for 1F & 2F (Streamlit, Pillow)
# - Windows 11, Python 3.11.9, VS Code
# - Uses OpenAI (optional), RapidFuzz (optional), requests (optional), Pillow, numpy, pandas, streamlit
# - Floor-map base images & aisle highlight PNGs are auto-overlaid based on search results
# - ✅ 仕様
#   1) ハイライト不透明度は 0.85 固定（スライダーなし）
#   2) フロアマップは常時表示（エクスパンダーなし）
#
# ▼ 必要パッケージ（PowerShell）
#   pip install -U pip
#   pip install streamlit pillow numpy pandas rapidfuzz requests python-dotenv "openai>=1.0.0"
#
# ▼ 起動
#   cd "C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app"
#   streamlit run shopping_assistant_streamlit.py

from __future__ import annotations

import os, re, sys, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, cast

import numpy as np
import pandas as pd
import streamlit as st

# ==== 環境 & OpenAI クライアント（任意） ====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT = None
EMBED_MODEL_DEFAULT = "text-embedding-3-small"
LLM_OPTIONS = ["gpt-5-mini", "gpt-4o-mini"]

try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        CLIENT = OpenAI()
except Exception:
    CLIENT = None

# ==== ローカル類似度（RapidFuzz 任意） ====
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib  # fallback
    HAVE_RAPIDFUZZ = False

# ==== 外部ヒント API（OFF/OBF/OPF 任意） ====
REQUESTS_OK = True
try:
    import requests
except Exception:
    REQUESTS_OK = False

def _strip_lang(tag: str) -> str:
    return tag.split(":", 1)[1] if ":" in tag else tag

def _fetch_json(url: str, params: Dict) -> Dict:
    if not REQUESTS_OK:
        return {}
    try:
        r = requests.get(
            url, params=params,
            headers={"User-Agent": "ShoppingAssistant/1.0 (+contact: you@example.com)"},
            timeout=6,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _off_search(q: str, page_size: int = 12) -> List[str]:
    js = _fetch_json(
        "https://world.openfoodfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    tags = set()
    for p in (js.get("products") or []):
        for t in (p.get("categories_tags") or []):
            tags.add(_strip_lang(t))
    return list(tags)

def _obf_search(q: str, page_size: int = 12) -> List[str]:
    js = _fetch_json(
        "https://world.openbeautyfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    tags = set()
    for p in (js.get("products") or []):
        for t in (p.get("categories_tags") or []):
            tags.add(_strip_lang(t))
    return list(tags)

def _opf_search(q: str, page_size: int = 12) -> List[str]:
    js = _fetch_json(
        "https://world.openproductsfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    tags = set()
    for p in (js.get("products") or []):
        for t in (p.get("categories_tags") or []):
            tags.add(_strip_lang(t))
    return list(tags)

@st.cache_data(show_spinner=False, ttl=3600)
def external_hints(q: str) -> Dict:
    if not REQUESTS_OK:
        return {"tags": [], "sources": {}}
    off, obf, opf = _off_search(q), _obf_search(q), _opf_search(q)
    merged, seen, out = [], set(), []
    for lst in (off, obf, opf):
        merged.extend(lst or [])
    for t in merged:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= 20:
            break
    return {"tags": out, "sources": {"off": off[:20], "obf": obf[:20], "opf": opf[:20]}}

# ==== フロアカテゴリ（OK 想定） ====  ※1Fに生鮮系の通路キーも追加
FLOOR1: Dict[str, List[str]] = {
    "1A": ["のり・わかめ・昆布", "豆・ごま・しいたけ", "春雨・切り干し大根"],
    "2A": ["つゆ・醤油・", "みりん・料理酒", "お酢・飯用酢"],
    "3A": ["マヨネーズ", "ドレッシング・たれ", "ケチャップ・ソース"],
    "4A": ["パスタ", "食用油"],
    "1B": ["だしの素", "ふりかけ・茶漬け", "味噌", "インスタント味噌汁"],
    "2B": ["缶詰", "砂糖・塩", "香辛料"],
    "3B": ["カレー・シチュー", "インスタント料理の素", "中華材料"],
    "4B": ["粉・パン粉", "製菓材料"],
    "5": ["インスタント袋麺", "インスタントカップ麺", "乾麺"],
    "6": ["スープ・コンソメ", "ジャム・蜂蜜", "シリアル・コーンフレーク", "紅茶・お茶・コーヒー"],
    "7": ["冷凍食品"],
    "8": ["冷凍食品", "アイスクリーム"],
    "9": ["チルド麺・漬物", "練製品", "こんにゃく", "豆腐・納豆"],
    "10": ["ヨーグルト", "バター・マーガリン", "チーズ", "牛乳・たまご"],
    "11": ["チルド飲料", "デザート", "ゼリー・プリン"],
    "12": ["ハム・ソーセージ", "チルド洋風・中華惣菜", "和風惣菜", "デザート・ゼリー・プリン"],
    "13": ["ミネラルウォーター", "清涼飲料", "栄養ドリンク", "酒割材"],
    "14": ["茶飲料", "野菜飲料", "果実飲料", "コーヒー・紅茶飲料"],
    "15": ["炭酸飲料", "乳酸飲料", "果実飲料"],
    "16": ["スナック菓子", "豆菓子・珍味", "季節菓子"],
    "17": ["チョコレート", "キャンディー", "ガム・キャラクター菓子", "クッキー・ビスケット"],
    "18": ["せんべい", "お茶菓子"],
    "19": ["食パン・菓子パン", "和洋生菓子"],
    # ▼ 生鮮系など（通路キーとして使う）
    "青果": ["リンゴ", "バナナ", "梨", "みかん", "スイカ", "パイナップル", "メロン", "シャインマスカット", "ぶどう", "イチゴ", "サクランボ", "オレンジ", "グレープフルーツ"],
    "野菜": ["キャベツ", "レタス", "大葉", "大根", "セロリ", "ピーマン", "白菜", "ニラ", "もやし", "きゅうり", "ごぼう", "ジャガイモ", "人参"],
    "鮮魚": ["マグロ", "シーフードミックス", "貝類", "わかめ", "サーモン", "鯛", "イカ", "タコ", "鯖", "しめ鯖", "アナゴ", "イワシ"],
    "精肉": ["豚バラ", "豚肩ロース", "牛肉", "鶏肉", "ひき肉", "合い挽き肉", "牛タン", "カルビ", "ハラミ", "豚トロ", "ヤゲン軟骨"],
    "お弁当": ["弁当", "かつ丼", "手巻き", "天丼"],
    "パン惣菜": ["パン", "食パン", "バターロール", "菓子パン", "ピザ", "バゲット", "ドーナツ類", "唐揚げ", "コロッケ", "惣菜類", "やきとり", "メンチカツ", "天ぷら"],
}
FLOOR2: Dict[str, List[str]] = {
    "1": ["ハミガキ・ハブラシ", "シャンプー・コンディショナー", "紙製品"],
    "2": ["石鹸", "入浴剤"],
    "3": ["タオル", "洗面・浴室用品", "ベビーフード・ベビー用品", "介護食"],
    "4": ["生理用品"],
    "5": ["毛染め", "女性用頭髪化粧品", "男性化粧品", "カミソリ"],
    "6": ["化粧小物", "制度化粧品"],
    "7": ["基礎化粧品", "季節化粧品", "洗顔・メイク落とし"],
    "8": ["婦人肌着・婦人靴下", "衣料品"],
    "9": ["衣料品", "紳士肌着・紳士靴下", "ベルト"],
    "10": ["寝具・インテリア", "靴用品"],
    "11": ["サンダル・スリッパ", "アスレチック", "生活電化"],
    "12": ["管球・電池", "時計"],
    "13": ["衣料用洗剤"],
    "14": ["おむつ", "選択用品"],
    "15": ["住居洗剤", "トイレ用品", "芳香剤", "虫ケア用品・カイロ"],
    "16": ["台所洗剤", "ラップ・ホイル", "炊事手袋・たわし・ふきん", "ごみ袋・ポリ袋"],
    "17": ["掃除用品", "掃除用品・収納用品", "アルミ成形品", "防虫・除湿剤"],
    "18": ["浄水器", "台所用品", "弁当小物", "割箸・紙コップ"],
    "19": ["製菓用品", "調理小物", "鍋・ケトル・フライパン", "台所用品"],
    "20": ["猫砂", "キャットフード"],
    "21": ["キャットフード", "ペット用品"],
    "22": ["ドックフード"],
    "23": ["ペット用品", "DIY用具・材料", "園芸用品", "障子紙・窓用品・安全用品"],
    "24": ["文具"],
    "25": ["事務用品"],
    "26": ["自転車", "自転車パーツ", "アウトドア用品"],
    "27": ["季節家電", "スポーツ", "傘", "上履き（バレーシューズ）"],
    "28": ["清酒"],
    "29": ["焼酎"],
    "30": ["ウイスキー・ブランデー", "スピリッツ・リキュール", "割材"],
    "31": ["缶チューハイ"],
    "32": ["缶チューハイ"],
    "33": ["ビール類"],
    "34": ["ビール類"],
}

# ==== 画像合成（Pillow） ====
from PIL import Image, ImageEnhance
try:
    from PIL.Image import Resampling  # Pillow ≥ 10
    RESAMPLE_LANCZOS = Resampling.LANCZOS
except Exception:
    def _pick_resample_lanczos():
        from PIL import Image as _Image
        for name in ("LANCZOS", "ANTIALIAS"):
            v = getattr(_Image, name, None)
            if v is not None:
                return v
        return 1  # fallback
    RESAMPLE_LANCZOS = _pick_resample_lanczos()

def pillow_overlay_image(base_rgba: Image.Image, ov_rgba: Image.Image,
                         x: int = 0, y: int = 0, scale: float = 1.0, alpha: float = 1.0) -> Image.Image:
    """別PNG(透過あり)を位置(x,y)に重ねる。scaleは倍率、alphaは全体不透明度。"""
    W2, H2 = ov_rgba.size
    new_size = (max(1, int(W2 * scale)), max(1, int(H2 * scale)))
    ov = ov_rgba.resize(new_size, RESAMPLE_LANCZOS).convert("RGBA")
    if alpha < 1.0:
        r, g, b, a = ov.split()
        a = ImageEnhance.Brightness(a).enhance(max(0.0, min(1.0, alpha)))
        ov = Image.merge("RGBA", (r, g, b, a))
    base = base_rgba.copy()
    base.alpha_composite(ov, (int(x), int(y)))
    return base

# ==== 正規化/インデックス ====
@dataclass(frozen=True)
class CategoryRow:
    floor: str
    aisle: str
    category: str
    alias: str

_SPLIT_RE = re.compile(r"[・/／,、\s]+")
def normalize_text(s: str) -> str:
    s2 = s.strip().lower()
    return s2.replace("　", "").replace(" ", "")

def expand_aliases(cat: str) -> List[str]:
    base = normalize_text(cat)
    parts = [normalize_text(p) for p in _SPLIT_RE.split(cat) if p]
    return list(dict.fromkeys([base] + parts))

def build_index() -> List[CategoryRow]:
    rows: List[CategoryRow] = []
    for aisle, cats in FLOOR1.items():
        for cat in cats:
            for alias in expand_aliases(cat):
                rows.append(CategoryRow("1F", aisle, cat, alias))
    for aisle, cats in FLOOR2.items():
        for cat in cats:
            for alias in expand_aliases(cat):
                rows.append(CategoryRow("2F", aisle, cat, alias))
    return rows

# ==== Embeddings ====
def embed_texts(texts: List[str], model: str) -> Optional[np.ndarray]:
    if CLIENT is None:
        return None
    try:
        resp = CLIENT.embeddings.create(model=model, input=texts)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vecs)
    except Exception:
        return None

def _unit_rows(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True); n[n == 0] = 1.0
    return mat / n

@st.cache_resource(show_spinner=False)
def cache_alias_embeddings(aliases_enriched: Tuple[str, ...], model: str, _key: str) -> Optional[np.ndarray]:
    return embed_texts(list(aliases_enriched), model)

# ==== LLM ヘルパ ====
def _try_parse_json(content: str) -> Optional[dict | list]:
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", content or "", flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

def llm_pick_best(item: str, shortlist: List[CategoryRow], model_name: str, hints: dict | None = None) -> Optional[dict]:
    if CLIENT is None:
        return None
    sys_prompt = (
        "You are a store category classifier.\n"
        "Prefer external category hints (Open Food/Beauty/Products Facts tags) when they clearly indicate the domain.\n"
        "Given a Japanese item and a shortlist (with floor/aisle), choose EXACTLY ONE best match.\n"
        "Interpret compound nouns correctly (e.g., 'ガムテープ' is packing tape, not candy). "
        "If an item contains household suffixes like テープ, ペン, 洗剤, 袋, ふきん, map to non-food.\n"
        "Return JSON: floor, aisle, category, confidence(0-100), explanation."
    )
    payload = {
        "item": item,
        "candidates": [{"floor": r.floor, "aisle": r.aisle, "category": r.category} for r in shortlist],
        "external_hints": (hints or {}),
    }
    try:
        resp = CLIENT.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        data = _try_parse_json(resp.choices[0].message.content or "")
        if isinstance(data, dict):
            return data
    except Exception:
        return None

def llm_pick_best_batch(items: List[str], shortlist_map: Dict[str, List[CategoryRow]], model_name: str,
                        hints_map: Dict[str, dict] | None = None) -> Optional[List[dict]]:
    if CLIENT is None:
        return None
    sys_prompt = (
        "You are a store category classifier.\n"
        "Prefer external category hints (Open Food/Beauty/Products Facts tags) when they clearly indicate the domain.\n"
        "For each item, pick EXACTLY ONE category from its shortlist (with floor/aisle). "
        "Handle compound nouns (e.g., 'ガムテープ' → packing tape). "
        "Return JSON array of {item, floor, aisle, category, confidence, explanation}."
    )
    batch_payload = []
    for it in items:
        cands = shortlist_map.get(it, [])
        batch_payload.append({
            "item": it,
            "candidates": [{"floor": r.floor, "aisle": r.aisle, "category": r.category} for r in cands],
            "external_hints": (hints_map or {}).get(it, {}),
        })
    try:
        resp = CLIENT.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": json.dumps(batch_payload, ensure_ascii=False)}],
        )
        data = _try_parse_json(resp.choices[0].message.content or "")
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return data["results"]
        if isinstance(data, list):
            return data
    except Exception:
        return None

# ==== Local Fuzzy ====
def fuzz_best(item: str, aliases: List[str]) -> Tuple[Optional[int], float, str]:
    q = normalize_text(item)
    if HAVE_RAPIDFUZZ:
        best = rf_process.extractOne(q, aliases, scorer=rf_fuzz.WRatio)
        if best:
            alias, sc, idx = best[0], float(best[1]), int(best[2])
            return idx, sc, alias
    else:
        try:
            close = difflib.get_close_matches(q, aliases, n=1, cutoff=0.5)  # type: ignore
            if close:
                a = close[0]; i = aliases.index(a)
                return i, 80.0, a
        except Exception:
            pass
    for i, a in enumerate(aliases):
        if a and (a in q or q in a):
            return i, 75.0, a
    return None, 0.0, ""

# ==== 通路ソート ====
_AISLE_RE = re.compile(r"^(\d+)([A-Za-zＡ-Ｚa-zａ-ｚ]*)?$")
_SUFFIX_ORDER = {"A": 0, "B": 1, "C": 2, "": 3}
def aisle_sort_key(s: str) -> Tuple[int, int, str]:
    if not isinstance(s, str) or not s:
        return (10**9, 10**9, "")
    m = _AISLE_RE.match(s)
    if not m:
        return (10**9, 10**9, s)
    n = int(m.group(1))
    suf_raw = (m.group(2) or "").strip()
    suf = suf_raw.upper() if suf_raw and suf_raw.isalpha() else ""
    return (n, _SUFFIX_ORDER.get(suf, 99), s)

# ==== 連携のためのパス設定（指定の絶対パス） ====
BASE_1F_PATH = r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\OK1階.png"
BASE_2F_PATH = r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\OK2階.png"
HIGHLIGHT_DIR_1F = r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\1階ハイライト"
HIGHLIGHT_DIR_2F = r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\2階ハイライト"

def _safe_open_rgba(path: str) -> Optional[Image.Image]:
    try:
        if path and os.path.exists(path) and str(path).lower().endswith(".png"):
            return Image.open(path).convert("RGBA")
    except Exception:
        return None
    return None

def _normalize_aisle_token(s: str) -> str:
    """例: '1a' -> '1A', '05b' -> '5B'。日本語通路名（青果 等）はそのまま。"""
    if not s:
        return ""
    s = str(s).strip()
    m = re.match(r"^0*(\d+)\s*([A-Za-z])?$", s)
    if not m:
        return s
    num = m.group(1)
    suf = (m.group(2) or "").upper()
    return f"{num}{suf}"

def _find_highlight_pngs_for_aisle(hl_dir: str, aisle: str) -> List[str]:
    """
    ハイライト用PNGの命名規則（想定）:
      - '<通路名>.png'（例: '1A.png', '5.png', '青果.png'）
    ディレクトリ内に該当ファイルがあれば全部重ねる（複数想定）
    """
    out: List[str] = []
    if not hl_dir or not os.path.isdir(hl_dir):
        return out
    target = _normalize_aisle_token(aisle)
    for name in os.listdir(hl_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        if _normalize_aisle_token(stem).upper() == target.upper():
            out.append(os.path.join(hl_dir, name))
    return out

# ==== UI ====
st.set_page_config(page_title="Shopping Assistant", layout="wide")
st.title("Shopping Assistant")

st.markdown("""
<style>
:root { --checkbox-nudge: -0.75em; }
.rowcell { height: 28px; display: flex; align-items: center; }
div[data-testid="stCheckbox"] { height: 28px; display: flex; align-items: center; transform: translateY(var(--checkbox-nudge)); }
/* minimal spacers */
.mt6 { height: 6px; }
.mt8 { height: 8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("設定")
    mode = st.radio("判定モード", ["AI (LLM + Embeddings)", "AI (Embeddings only)", "Local (RapidFuzz)"], index=0)
    embed_model = st.selectbox("Embeddings モデル", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    chat_model = st.selectbox("LLM モデル", LLM_OPTIONS, index=0)

    st.markdown("<div class='mt6'></div>", unsafe_allow_html=True)
    batch_llm = st.checkbox("LLM まとめて1回判定（高速）", value=True)

    topk = st.slider("候補K（Embeddingsで絞り込み）", 5, 20, 10)
    embed_conf_threshold = st.slider("埋め込みだけで確定（cos類似の閾値）", 0.80, 0.98, 0.92, 0.01)

    use_external = st.checkbox("無料API（OFF/OBF/OPF）でカテゴリ候補を補強する", value=True and REQUESTS_OK)
    if not REQUESTS_OK:
        st.caption("requests が見つかりません。`pip install requests` で有効化できます。")
    st.caption(f"APIキー: {'✅ 検出' if OPENAI_API_KEY else '❌ 未設定（.envにOPENAI_API_KEY）'}")

    st.divider()
    st.subheader("入力例")
    sample = "ガムテープ\nリンゴジュース\nティッシュペーパー\nボールペン\nはちみつ\nカップ麺\n猫砂\n缶ビール"
    if st.button("例を貼る"):
        st.session_state["items_text"] = sample

items_text = st.text_area(
    "買い物リスト（1行に1品）",
    value=st.session_state.get("items_text", ""),
    height=180,
    placeholder="例）\nガムテープ\nリンゴジュース\n…",
)

def reset_checks() -> None:
    for k in list(st.session_state.keys()):
        if str(k).startswith("done_"):
            del st.session_state[k]

def clear_all() -> None:
    reset_checks()
    st.session_state["items_text"] = ""
    st.session_state["__last_results__"] = None
    st.session_state["__last_details__"] = None

run = st.button("検索", type="primary")

# ---- build index & pre-cache ----
def build_rows_and_aliases():
    rows = build_index()
    aliases = [r.alias for r in rows]
    enriched = tuple(f"{r.alias} | {r.category} | {r.floor}{r.aisle}" for r in rows)
    return rows, aliases, enriched

rows, aliases, enriched = build_rows_and_aliases()
_ = cache_alias_embeddings(enriched, EMBED_MODEL_DEFAULT, "warmup") if CLIENT is not None else None

def shortlist_many_with_embeddings(queries: List[str], k: int, model: str):
    if CLIENT is None:
        return {q: (rows[:k], 0.0) for q in queries}
    qvecs = embed_texts([normalize_text(q) for q in queries], model)
    base_vecs = cache_alias_embeddings(enriched, model, f"key-{model}")
    if qvecs is None or base_vecs is None:
        return {q: (rows[:k], 0.0) for q in queries}
    alias_unit = _unit_rows(base_vecs)
    out = {}
    for qi, qv in enumerate(qvecs):
        q_unit = qv / (np.linalg.norm(qv) + 1e-9)
        sims = alias_unit @ q_unit
        top_idx = np.argsort(-sims)[:k]
        shortlist = [rows[int(i)] for i in top_idx]
        top_sim = float(sims[int(top_idx[0])]) if len(top_idx) > 0 else 0.0
        out[queries[qi]] = (shortlist, top_sim)
    return out

# ---- run match ----
if run:
    items = [s for s in (items_text or "").splitlines() if s.strip()]
    if not items:
        st.warning("品名を入力してください。")
        st.stop()

    reset_checks()
    results: List[Dict[str, object]] = []
    details: List[Tuple[str, List[Dict[str, str]]]] = []

    if mode == "AI (LLM + Embeddings)" and CLIENT is not None:
        with st.spinner("Embeddings で候補抽出中…"):
            sl_map = shortlist_many_with_embeddings(items, topk, embed_model)

        decided: Dict[str, Optional[dict]] = {}
        for it, (sl, top_sim) in sl_map.items():
            if top_sim >= embed_conf_threshold:
                t = sl[0]
                decided[it] = {"floor": t.floor, "aisle": t.aisle, "category": t.category,
                               "confidence": int(round(top_sim * 100)), "explanation": "embedding confident"}
            else:
                decided[it] = None

        remaining = [it for it in items if decided[it] is None]
        hints_map: Dict[str, dict] = {}
        if use_external and remaining:
            with st.spinner("無料APIからカテゴリヒント取得中…"):
                for it in remaining:
                    hints_map[it] = external_hints(it)

        if remaining and batch_llm:
            with st.spinner("LLM まとめ判定中…"):
                sub_map = {it: sl_map[it][0] for it in remaining}
                decided_list = llm_pick_best_batch(remaining, sub_map, chat_model, hints_map=hints_map) or []
                mapped = {d["item"]: d for d in decided_list if isinstance(d, dict) and d.get("item")}
                for it in remaining:
                    pick = mapped.get(it)
                    if not pick:
                        t = sl_map[it][0][0]
                        pick = {"floor": t.floor, "aisle": t.aisle, "category": t.category,
                                "confidence": 70, "explanation": "fallback: embedding top"}
                    decided[it] = pick
        elif remaining:
            with st.spinner("LLM 判定中…"):
                for it in remaining:
                    sl = sl_map[it][0]
                    pick = llm_pick_best(it, sl, chat_model, hints=hints_map.get(it))
                    if not pick:
                        t = sl[0]
                        pick = {"floor": t.floor, "aisle": t.aisle, "category": t.category,
                                "confidence": 70, "explanation": "fallback: embedding top"}
                    decided[it] = pick

        for it in items:
            pick = decided[it] or {}
            results.append({
                "品名": it, "階": pick.get("floor", ""), "通路": pick.get("aisle", ""),
                "カテゴリ(推定)": pick.get("category", ""), "score": int(round(float(pick.get("confidence", 0)))),
                "explanation": pick.get("explanation", ""),
            })
            details.append((it, [{"候補カテゴリ": r.category, "floor/aisle": f"{r.floor}{r.aisle}"} for r in sl_map[it][0]]))

    elif mode == "AI (Embeddings only)" and CLIENT is not None:
        with st.spinner("Embeddings 判定中…"):
            sl_map = shortlist_many_with_embeddings(items, k=1, model=embed_model)
        for it in items:
            t = sl_map[it][0][0]
            results.append({"品名": it, "階": t.floor, "通路": t.aisle, "カテゴリ(推定)": t.category,
                            "score": int(round(sl_map[it][1] * 100)), "explanation": "embedding only"})
            details.append((it, [{"候補カテゴリ": t.category, "floor/aisle": f"{t.floor}{t.aisle}"}]))

    else:
        for it in items:
            q = normalize_text(it)
            if HAVE_RAPIDFUZZ:
                best = rf_process.extractOne(q, [r.alias for r in rows], scorer=rf_fuzz.WRatio)
                if best:
                    idx = int(best[2]); sc = float(best[1]); r = rows[idx]
                    results.append({"品名": it, "階": r.floor, "通路": r.aisle, "カテゴリ(推定)": r.category,
                                    "score": int(round(sc)), "explanation": "local"})
                    continue
            else:
                close = difflib.get_close_matches(q, [r.alias for r in rows], n=1, cutoff=0.5)  # type: ignore
                if close:
                    a = close[0]; i = [r.alias for r in rows].index(a); r = rows[i]
                    results.append({"品名": it, "階": r.floor, "通路": r.aisle, "カテゴリ(推定)": r.category,
                                    "score": 80, "explanation": "local"})
                    continue
            results.append({"品名": it, "階": "", "通路": "", "カテゴリ(推定)": "(未判定)", "score": 0, "explanation": "not found"})

    # sort & numbering
    df = pd.DataFrame(results)
    if not df.empty:
        def _as_key(x):
            m = re.match(r"^(\d+)([A-Za-zＡ-Ｚa-zａ-ｚ]*)?$", str(x)) if x else None
            if not m:
                return (10**9, 10**9, str(x))
            n = int(m.group(1)); suf = (m.group(2) or "").upper()
            order = {"A": 0, "B": 1, "C": 2, "": 3}.get(suf if suf.isalpha() else "", 99)
            return (n, order, str(x))
        df["_floor_ord"] = df["階"].map(lambda f: 0 if f == "1F" else (1 if f == "2F" else 2))
        df["_a_n"] = df["通路"].map(lambda x: _as_key(x)[0] if x else 10**9)
        df["_a_s"] = df["通路"].map(lambda x: _as_key(x)[1] if x else 10**9)
        df_sorted = (
            df.sort_values(by=["_floor_ord", "_a_n", "_a_s", "通路"], ascending=[True, True, True, True])
              .drop(columns=["_floor_ord", "_a_n", "_a_s"])
        )
        df_sorted.insert(0, "No.", range(1, len(df_sorted) + 1))
    else:
        df_sorted = df

    st.session_state["__last_results__"] = df_sorted.copy()
    st.session_state["__last_details__"] = details

# ---- render results ----
df_to_show = cast(Optional[pd.DataFrame], st.session_state.get("__last_results__"))
details = cast(List[Tuple[str, List[Dict[str, str]]]], st.session_state.get("__last_details__") or [])

if df_to_show is not None and not df_to_show.empty:
    st.markdown("<div class='mt8'></div>", unsafe_allow_html=True)
    st.subheader("チェックリスト")

    hcols = st.columns([0.08, 0.08, 0.34, 0.10, 0.12, 0.24, 0.10])
    for col, name in zip(hcols, ["✓", "No.", "品名", "階", "通路", "カテゴリ(推定)", "score"]):
        col.markdown(f"<div class='rowcell'><b>{name}</b></div>", unsafe_allow_html=True)

    def greyspan(text: str, done: bool) -> str:
        style = "color:#9aa0a6; opacity:0.65" if done else ""
        return f"<div class='rowcell'><span style='{style}'>{text}</span></div>"

    prev_floor: Optional[str] = None
    for _, row in df_to_show.iterrows():
        if prev_floor is not None and prev_floor != row["階"]:
            st.markdown("<div style='border-top:2px dashed #c7c7c7; margin:6px 0;'></div>", unsafe_allow_html=True)
        prev_floor = row["階"]  # type: ignore[index]

        no = int(row["No."])  # type: ignore[index]
        key = f"done_{no}"
        checked = bool(st.session_state.get(key, False))
        cols = st.columns([0.08, 0.08, 0.34, 0.10, 0.12, 0.24, 0.10])

        cols[0].checkbox(" ", value=checked, key=key, label_visibility="collapsed")
        cols[1].markdown(greyspan(str(no), st.session_state[key]), unsafe_allow_html=True)
        cols[2].markdown(greyspan(str(row["品名"]), st.session_state[key]), unsafe_allow_html=True)
        cols[3].markdown(greyspan(str(row["階"]), st.session_state[key]), unsafe_allow_html=True)
        cols[4].markdown(greyspan(str(row["通路"]), st.session_state[key]), unsafe_allow_html=True)
        cols[5].markdown(greyspan(str(row["カテゴリ(推定)"]), st.session_state[key]), unsafe_allow_html=True)
        cols[6].markdown(greyspan(str(row["score"]), st.session_state[key]), unsafe_allow_html=True)
else:
    st.info("「検索」を押すと結果が表示されます。")

# ==== チェック状況に応じた「ハイライト対象通路」の算出 ====
def compute_active_aisles(df_sorted: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """
    通路ごとの全アイテムがチェック済みなら、その通路のハイライトを消す。
    そうでない通路だけを「アクティブ」として返す。
    return: {"1F": [通路...], "2F": [通路...]}
    """
    active: Dict[str, List[str]] = {"1F": [], "2F": []}
    if df_sorted is None or df_sorted.empty:
        return active

    # 欠損/空文字を排除して必要列のみ取り出し
    valid = df_sorted[(df_sorted["通路"].notna()) & (df_sorted["通路"].astype(str).str.strip() != "")]
    if valid.empty:
        return active
    valid = valid.loc[:, ["階", "通路", "No."]].copy()

    # ★ Pylanceのタプル展開誤検知を避けるため、まず list に畳んでから for で回す
    grp = (
        valid.groupby(["階", "通路"], dropna=True)["No."]
             .apply(list)
             .reset_index(name="nos")
    )

    for _, row in grp.iterrows():
        floor = str(row["階"])
        aisle = str(row["通路"])
        nos_list = row["nos"] if isinstance(row["nos"], list) else [row["nos"]]
        # No. は int 想定だが、念のため int キャストしてキー生成
        all_checked = all(bool(st.session_state.get(f"done_{int(no)}", False)) for no in nos_list)
        if not all_checked:
            active.setdefault(floor, []).append(aisle)

    # 重複除去 & ソート（視覚的安定性）
    for fl in ("1F", "2F"):
        uniq = sorted(set(active.get(fl, [])),
                      key=lambda a: (aisle_sort_key(a)[0], aisle_sort_key(a)[1], str(a)))
        active[fl] = uniq
    return active

# ==== フロアマップ（1F/2F 自動ハイライト：常時表示） ====
ALPHA_OVERLAY_1F = 0.85
ALPHA_OVERLAY_2F = 0.85

def render_floor_maps_with_auto_overlay(results_df: Optional[pd.DataFrame],
                                        force_active_aisles: Optional[Dict[str, List[str]]] = None) -> None:
    """results_df から各フロアの通路を集計し、該当PNGを重ねて可視化。"""
    c1, c2 = st.columns(2)
    shown_any = False

    # --- 1F ---
    base1 = _safe_open_rgba(BASE_1F_PATH)
    if base1 is not None:
        if force_active_aisles is not None:
            aisles_1f: List[str] = force_active_aisles.get("1F", [])
        else:
            aisles_1f = []
            if results_df is not None and not results_df.empty:
                aisles_1f = [str(a) for a in results_df[results_df["階"] == "1F"]["通路"].dropna().unique().tolist()]
        merged1 = base1
        highlighted_1f: List[str] = []
        if aisles_1f:
            for a in aisles_1f:
                for png_path in _find_highlight_pngs_for_aisle(HIGHLIGHT_DIR_1F, a):
                    ov = _safe_open_rgba(png_path)
                    if ov is not None:
                        merged1 = pillow_overlay_image(merged1, ov, 0, 0, 1.0, ALPHA_OVERLAY_1F)
                        highlighted_1f.append(os.path.basename(png_path))
        with c1:
            cap = "1F レイアウト"
            if highlighted_1f:
                cap += f"（ハイライト: {', '.join(sorted(set([_normalize_aisle_token(os.path.splitext(h)[0]) for h in highlighted_1f])))}）"
            else:
                cap += "（ハイライト無し or 全チェック完了）"
            st.image(np.array(merged1), caption=cap, use_container_width=True)
            shown_any = True
    else:
        with c1:
            st.info("1F ベース画像が見つかりません。パスを確認してください。")

    # --- 2F ---
    base2 = _safe_open_rgba(BASE_2F_PATH)
    if base2 is not None:
        if force_active_aisles is not None:
            aisles_2f: List[str] = force_active_aisles.get("2F", [])
        else:
            aisles_2f = []
            if results_df is not None and not results_df.empty:
                aisles_2f = [str(a) for a in results_df[results_df["階"] == "2F"]["通路"].dropna().unique().tolist()]
        merged2 = base2
        highlighted_2f: List[str] = []
        if aisles_2f:
            for a in aisles_2f:
                for png_path in _find_highlight_pngs_for_aisle(HIGHLIGHT_DIR_2F, a):
                    ov = _safe_open_rgba(png_path)
                    if ov is not None:
                        merged2 = pillow_overlay_image(merged2, ov, 0, 0, 1.0, ALPHA_OVERLAY_2F)
                        highlighted_2f.append(os.path.basename(png_path))
        with c2:
            cap2 = "2F レイアウト"
            if highlighted_2f:
                cap2 += f"（ハイライト: {', '.join(sorted(set([_normalize_aisle_token(os.path.splitext(h)[0]) for h in highlighted_2f])))}）"
            else:
                cap2 += "（ハイライト無し or 全チェック完了）"
            st.image(np.array(merged2), caption=cap2, use_container_width=True)
            shown_any = True
    else:
        with c2:
            st.info("2F ベース画像が見つかりません。パスを確認してください。")

    if not shown_any:
        st.warning("ベース画像が両方見つかりません。指定パスのPNGを配置してください。")

# 常時表示
st.markdown("### フロアマップ（1F / 2F）— 自動ハイライト")
df_for_overlay = cast(Optional[pd.DataFrame], st.session_state.get("__last_results__"))
active_aisles = compute_active_aisles(df_for_overlay)
render_floor_maps_with_auto_overlay(df_for_overlay, force_active_aisles=active_aisles)

# ==== 索引・候補 ====
with st.expander("カテゴリ索引（確認用）"):
    idx_df = pd.DataFrame([{"floor": r.floor, "aisle": r.aisle, "category": r.category} for r in build_index()]).drop_duplicates()
    st.dataframe(idx_df.reset_index(drop=True), use_container_width=True)

with st.expander("候補リスト（参照）"):
    if details:
        for it, lst in details:
            st.markdown(f"**{it}**")
            dd = pd.DataFrame(lst) if lst else pd.DataFrame()
            st.dataframe(dd, use_container_width=True)
    else:
        st.caption("まだ候補はありません。検索後に表示されます。")

st.button("🧹 全てクリア", on_click=clear_all, type="secondary")
