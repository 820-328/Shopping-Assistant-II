import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance

from api_client import get_openai_client, external_hints
from constants import (
    CategoryRow,
    FLOOR1,
    FLOOR2,
    SYNONYMS,
    ALPHA_OVERLAY_1F,
    ALPHA_OVERLAY_2F,
    CATEGORY_ALIASES,
)

# ========== 文字正規化・別名展開・通路ソート ==========
_SPLIT_RE = re.compile(r"[・/／,、\s]+")
_AISLE_RE = re.compile(r"^(\d+)([A-Za-zＡ-Ｚa-zａ-ｚ]*)?$")
_SUFFIX_ORDER = {"A": 0, "B": 1, "C": 2, "": 3}

def normalize_text(s: str) -> str:
    s2 = s.strip().lower()
    return s2.replace("　", "").replace(" ", "")

def _SPILT_SAFE(cat: str) -> List[str]:
    try:
        return [p for p in _SPLIT_RE.split(cat) if p]
    except Exception:
        return [cat]

CATEGORY_ALIAS_LOOKUP = {
    normalize_text(alias): category
    for category, aliases in CATEGORY_ALIASES.items()
    for alias in aliases
}

CATEGORY_BOOST_KEYWORDS: Dict[str, List[str]] = {
    "DIY用品・工具": ["テープ", "養生", "梱包", "粘着", "接着", "貼付", "補修", "ボンド"],
    "文房具・事務用品": ["ペン", "シャープ", "ノート", "事務", "文房具", "ステーショナリー", "メモ", "ホチキス"],
    "ペット用品": ["ペット", "犬", "猫", "インコ", "小鳥", "ペット用", "ペットグッズ"],
    "キャットフード": ["猫", "キャット", "インコ", "鳥", "餌", "ペットフード", "ごはん"],
    "猫砂": ["猫砂", "ネコ砂", "トイレ砂", "猫トイレ"],
    "果実飲料": ["ジュース", "果汁", "リンゴ", "アップル", "オレンジ", "グレープ", "マンゴー", "スムージー"],
    "茶飲料": ["茶", "ティー", "緑茶", "麦茶", "ほうじ茶", "ウーロン", "ジャスミン"],
    "炭酸飲料": ["炭酸", "サイダー", "コーラ", "ソーダ", "ラムネ", "スパークリング"],
    "紙製品": ["ティッシュ", "ペーパー", "ナプキン", "トイレット"],
}

CATEGORY_NEGATIVE_KEYWORDS: Dict[str, List[str]] = {
    "ガム・キャラクター菓子": ["テープ", "養生", "梱包", "粘着", "接着", "貼付", "工具"],
    "チョコレート": ["テープ", "養生", "梱包", "粘着"],
    "クッキー・ビスケット": ["テープ", "養生", "梱包"],
    "キャンディー": ["テープ", "養生", "梱包"],
    "鮮魚": ["餌", "エサ", "ペット", "鳥", "インコ", "キャット", "ドッグ"],
}

LOCAL_MATCH_PRIORITY = {
    "synonym": 0,
    "exact": 1,
    "substring": 2,
    "fuzzy": 3,
}

def _resolve_synonym(word: str) -> str:
    if not word:
        return word
    base = SYNONYMS.get(word)
    if base:
        return base
    norm = normalize_text(word)
    if norm in CATEGORY_ALIAS_LOOKUP:
        return CATEGORY_ALIAS_LOOKUP[norm]
    for alias_norm, category in CATEGORY_ALIAS_LOOKUP.items():
        if alias_norm and alias_norm in norm:
            return category
    return word

def expand_aliases(cat: str) -> List[str]:
    base = normalize_text(cat)
    parts = [normalize_text(p) for p in _SPILT_SAFE(cat)]
    extras = CATEGORY_ALIASES.get(cat, [])
    if extras:
        parts.extend(normalize_text(p) for p in extras)
    return list(dict.fromkeys([base] + parts))

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

# ========== 索引 ==========
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

# ========== Embeddings ==========
@st.cache_resource(show_spinner=False)
def _cache_alias_embeddings(aliases_enriched: Tuple[str, ...], model: str, _key: str) -> Optional[np.ndarray]:
    return embed_texts(list(aliases_enriched), model)

def embed_texts(texts: List[str], model: str) -> Optional[np.ndarray]:
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.embeddings.create(model=model, input=texts)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vecs)
    except Exception:
        return None

def _unit_rows(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n

def shortlist_many_with_embeddings(
    queries: List[str], k: int, model: str, rows: List[CategoryRow], enriched: Tuple[str, ...]
):
    client = get_openai_client()
    # Local fallback using RapidFuzz/difflib when OpenAI client or embeddings are unavailable
    def _local_shortlist(q: str):
        aliases = [r.alias for r in rows]
        qn = normalize_text(_resolve_synonym(q))
        try:
            if HAVE_RAPIDFUZZ:
                bests = rf_process.extract(qn, aliases, scorer=rf_fuzz.WRatio, limit=k)
                pairs = [(rows[int(b[2])], float(b[1]) / 100.0) for b in bests]
                top_sim = pairs[0][1] if pairs else 0.0
                return (pairs if pairs else [(rows[0], 0.0)], top_sim)
            else:
                import difflib
                matches = difflib.get_close_matches(qn, aliases, n=k, cutoff=0.0)  # type: ignore
                pairs = []
                for a in matches:
                    try:
                        idx = aliases.index(a)
                    except Exception:
                        continue
                    pairs.append((rows[idx], 0.7))
                return (pairs if pairs else [(rows[0], 0.0)], 0.0)
        except Exception:
            return ([(rows[0], 0.0)], 0.0)

    if client is None:
        return {q: _local_shortlist(q) for q in queries}
    q_norm = [normalize_text(_resolve_synonym(q)) for q in queries]
    qvecs = embed_texts(q_norm, model)
    base_vecs = _cache_alias_embeddings(enriched, model, f"key-{model}")
    if qvecs is None or base_vecs is None:
        return {q: _local_shortlist(q) for q in queries}
    alias_unit = _unit_rows(base_vecs)
    out = {}
    for qi, qv in enumerate(qvecs):
        q_unit = qv / (np.linalg.norm(qv) + 1e-9)
        sims = alias_unit @ q_unit
        top_idx = np.argsort(-sims)[:k]
        shortlist = []
        for idx in top_idx:
            shortlist.append((rows[int(idx)], float(sims[int(idx)])))
        top_sim = float(sims[int(top_idx[0])]) if len(top_idx) > 0 else 0.0
        if not shortlist:
            fallback_pairs, top_sim = _local_shortlist(queries[qi])
            shortlist = fallback_pairs
        out[queries[qi]] = (shortlist, top_sim)
    return out

# ========== Local candidate helpers ==========

def _local_candidates_for_item(
    item: str,
    rows: List[CategoryRow],
    aliases: List[str],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    q_raw = (item or "").strip()
    q_norm = normalize_text(q_raw)
    results: List[Dict[str, Any]] = []
    seen = set()

    def add(row: CategoryRow, alias: str, score: float, match_type: str) -> None:
        key = (row.floor, row.aisle, row.category, alias)
        if key in seen:
            return
        seen.add(key)
        alias_norm = normalize_text(alias) if alias else normalize_text(row.alias)
        score_adj = float(score)

        for boost in CATEGORY_BOOST_KEYWORDS.get(row.category, []):
            nb = normalize_text(boost)
            if nb and nb in q_norm:
                score_adj += 15.0

        for negate in CATEGORY_NEGATIVE_KEYWORDS.get(row.category, []):
            nn = normalize_text(negate)
            if nn and (nn in q_norm or (alias_norm and nn in alias_norm)):
                score_adj -= 40.0

        score_adj = max(0.0, min(score_adj, 100.0))

        results.append({
            "row": row,
            "alias": alias,
            "score": score_adj,
            "match_type": match_type,
        })

    synonym_cat = _resolve_synonym(q_raw)
    if synonym_cat and synonym_cat != q_raw:
        for row in rows:
            if row.category == synonym_cat:
                add(row, row.alias, 100.0, "synonym")
                break

    if HAVE_RAPIDFUZZ and aliases:
        matches = rf_process.extract(q_norm, aliases, scorer=rf_fuzz.WRatio, limit=limit)
        for alias, score, idx in matches:
            row = rows[int(idx)]
            alias_norm = alias
            if alias_norm == q_norm:
                mtype = "exact"
            elif alias_norm and (alias_norm in q_norm or q_norm in alias_norm):
                mtype = "substring"
            else:
                mtype = "fuzzy"
            add(row, alias_norm, float(score), mtype)
    else:
        try:
            import difflib

            matches = difflib.get_close_matches(q_norm, aliases, n=limit, cutoff=0.0)  # type: ignore[arg-type]
            for alias in matches:
                idx = aliases.index(alias)
                row = rows[idx]
                add(row, alias, 80.0, "fuzzy")
        except Exception:
            pass

    results.sort(key=lambda c: (LOCAL_MATCH_PRIORITY.get(c["match_type"], 4), -c["score"]))
    return results


def _serialize_candidates(candidates: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in candidates[:limit]:
        row: CategoryRow = entry["row"]
        out.append({
            "category": row.category,
            "floor": row.floor,
            "aisle": row.aisle,
            "alias": entry.get("alias", row.alias),
            "score": round(float(entry.get("score", 0.0)), 1),
            "match_type": entry.get("match_type", "fuzzy"),
        })
    return out


def _find_row_by_category(rows: List[CategoryRow], category: str, floor: Optional[str] = None, aisle: Optional[str] = None) -> Optional[CategoryRow]:
    for row in rows:
        if row.category != category:
            continue
        if floor and row.floor != floor:
            continue
        if aisle and row.aisle != aisle:
            continue
        return row
    return None


# ========== LLM ==========
# ========== LLM ==========
def _try_parse_json(content: str):
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
    client = get_openai_client()
    if client is None:
        return None
    sys_prompt = (
        "You are a store category classifier.\n"
        "Prefer external category hints (Open Food/Beauty/Products Facts tags) when they clearly indicate the domain.\n"
        "Given a Japanese item and a shortlist (with floor/aisle), choose EXACTLY ONE best match.\n"
        "Guidelines:\n"
        "- 'テープ','養生','梱包','粘着','接着' imply DIY/工具 categories. Never map them to confectionery.\n"
        "- 'インコ','鳥','ペット','猫','犬' imply pet supplies. Never map them to鮮魚 or fresh food.\n"
        "- Beverage terms like ジュース/果汁/茶/炭酸/乳酸菌 should stay in飲料 categories.\n"
        "Interpret compound nouns correctly (e.g., 'ガムテープ' is packing tape, not candy).\n"
        "Return JSON: floor, aisle, category, confidence(0-100), explanation."
    )
    payload = {
        "item": item,
        "candidates": [{"floor": r.floor, "aisle": r.aisle, "category": r.category} for r in shortlist],
        "external_hints": (hints or {}),
    }
    try:
        resp = client.chat.completions.create(
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
    client = get_openai_client()
    if client is None:
        return None
    sys_prompt = (
        "You are a store category classifier.\n"
        "Prefer external category hints (Open Food/Beauty/Products Facts tags) when they clearly indicate the domain.\n"
        "For each item, pick EXACTLY ONE category from its shortlist (with floor/aisle).\n"
        "Guidelines:\n"
        "- 'テープ','養生','梱包','粘着','接着' imply DIY/工具 categories. Never map them to confectionery.\n"
        "- 'インコ','鳥','ペット','猫','犬' imply pet supplies. Avoid鮮魚 or fresh food.\n"
        "- ジュース/果汁/茶/炭酸/乳酸菌など飲料語は飲料カテゴリを優先。\n"
        "Interpret compound nouns correctly (e.g., 'ガムテープ' is packing tape).\n"
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
        resp = client.chat.completions.create(
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

# ========== Local Fuzzy ==========
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    import difflib  # fallback
    HAVE_RAPIDFUZZ = False

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
                a = close[0]
                i = aliases.index(a)
                return i, 80.0, a
        except Exception:
            pass
    for i, a in enumerate(aliases):
        if a and (a in q or q in a):
            return i, 75.0, a
    return None, 0.0, ""

# ========== 分類メイン ==========
def classify_items(
    items: List[str],
    mode: str,
    embed_model: str,
    chat_model: str,
    batch_llm: bool,
    topk: int,
    embed_conf_threshold: float,
    use_external: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    rows = build_index()
    aliases = [r.alias for r in rows]
    enriched = tuple(f"{r.alias} | {r.category} | {r.floor}{r.aisle}" for r in rows)

    local_map = {
        it: _local_candidates_for_item(it, rows, aliases, limit=max(6, topk))
        for it in items
    }
    direct_map: Dict[str, List[Dict[str, Any]]] = {}
    for it, cand in local_map.items():
        direct = [c for c in cand if c["match_type"] in ("synonym", "exact", "substring")]
        if direct:
            direct_map[it] = direct

    results: List[Dict[str, Any]] = []
    detail_records: List[Dict[str, Any]] = []

    client = get_openai_client()
    if mode.startswith("AI") and client is None:
        mode = "Local (RapidFuzz)"

    needs_ai = [it for it in items if it not in direct_map and mode.startswith("AI")]
    ai_decisions: Dict[str, Dict[str, Any]] = {}

    if mode == "AI (LLM + Embeddings)" and client is not None and needs_ai:
        sl_map = shortlist_many_with_embeddings(needs_ai, max(topk, 3), embed_model, rows, enriched)
        decided: Dict[str, Optional[Dict[str, Any]]] = {}
        for it, (pairs, top_sim) in sl_map.items():
            if pairs and top_sim >= embed_conf_threshold:
                row, score = pairs[0]
                decided[it] = {
                    "pick_row": row,
                    "confidence": int(round(max(0.0, min(score, 1.0)) * 100)),
                    "explanation": "embedding confident",
                    "candidates": pairs,
                }
            else:
                decided[it] = None

        remaining = [it for it in needs_ai if decided[it] is None]
        hints_map: Dict[str, dict] = {}
        if use_external and remaining:
            hints_map = {it: external_hints(it) for it in remaining}

        if remaining and batch_llm:
            sub_map = {it: [row for row, _ in sl_map[it][0]] for it in remaining}
            decided_list = llm_pick_best_batch(remaining, sub_map, chat_model, hints_map=hints_map) or []
            mapped = {d.get("item"): d for d in decided_list if isinstance(d, dict) and d.get("item")}
            for it in remaining:
                pick = mapped.get(it)
                if pick:
                    row = _find_row_by_category(rows, pick.get("category", ""), pick.get("floor"), pick.get("aisle"))
                    if row is None and sl_map[it][0]:
                        row = sl_map[it][0][0][0]
                    confidence = int(round(float(pick.get("confidence", 80))))
                    decided[it] = {
                        "pick_row": row,
                        "confidence": confidence,
                        "explanation": pick.get("explanation", "llm selection"),
                        "candidates": sl_map[it][0],
                    }
                else:
                    pairs = sl_map[it][0]
                    if pairs:
                        row, score = pairs[0]
                        decided[it] = {
                            "pick_row": row,
                            "confidence": int(round(max(0.0, min(score, 1.0)) * 100)),
                            "explanation": "fallback: embedding top",
                            "candidates": pairs,
                        }
        elif remaining:
            for it in remaining:
                pairs = sl_map[it][0]
                pick = llm_pick_best(it, [row for row, _ in pairs], chat_model, hints=hints_map.get(it))
                if pick:
                    row = _find_row_by_category(rows, pick.get("category", ""), pick.get("floor"), pick.get("aisle"))
                    if row is None and pairs:
                        row = pairs[0][0]
                    confidence = int(round(float(pick.get("confidence", 80))))
                    decided[it] = {
                        "pick_row": row,
                        "confidence": confidence,
                        "explanation": pick.get("explanation", "llm selection"),
                        "candidates": pairs,
                    }
                elif pairs:
                    row, score = pairs[0]
                    decided[it] = {
                        "pick_row": row,
                        "confidence": int(round(max(0.0, min(score, 1.0)) * 100)),
                        "explanation": "fallback: embedding top",
                        "candidates": pairs,
                    }
        for it, info in decided.items():
            if info is None and sl_map.get(it):
                row, score = sl_map[it][0][0]
                info = {
                    "pick_row": row,
                    "confidence": int(round(max(0.0, min(score, 1.0)) * 100)),
                    "explanation": "fallback: embedding top",
                    "candidates": sl_map[it][0],
                }
            if info:
                ai_decisions[it] = info

    elif mode == "AI (Embeddings only)" and client is not None and needs_ai:
        sl_map = shortlist_many_with_embeddings(needs_ai, max(topk, 3), embed_model, rows, enriched)
        for it, (pairs, _) in sl_map.items():
            if not pairs:
                continue
            row, score = pairs[0]
            ai_decisions[it] = {
                "pick_row": row,
                "confidence": int(round(max(0.0, min(score, 1.0)) * 100)),
                "explanation": "embedding only",
                "candidates": pairs,
            }

    for it in items:
        local_candidates = local_map.get(it, [])
        direct_candidates = [c for c in direct_map.get(it, []) if c["match_type"] in ("synonym", "exact", "substring")]

        detail_record: Dict[str, Any] = {
            "item": it,
            "local": _serialize_candidates(direct_candidates if direct_candidates else local_candidates[:3]),
            "ai": [],
        }

        chosen_row: Optional[CategoryRow] = None
        confidence = 0
        explanation = ""

        if direct_candidates:
            primary = direct_candidates[0]
            chosen_row = primary["row"]
            confidence = int(round(min(primary.get("score", 100.0), 100.0)))
            explanation = {
                "synonym": "local synonym match",
                "exact": "local exact match",
                "substring": "local partial match",
            }.get(primary["match_type"], "local match")
        elif it in ai_decisions:
            info = ai_decisions[it]
            chosen_row = info.get("pick_row")
            confidence = int(info.get("confidence", 70))
            explanation = info.get("explanation", "ai selection")
            ai_candidates = info.get("candidates", [])
            ai_entries = [
                {"row": row, "alias": row.alias, "score": float(score) * 100.0, "match_type": "ai"}
                for row, score in ai_candidates
            ]
            detail_record["ai"] = _serialize_candidates(ai_entries, limit=5)
        elif local_candidates:
            fallback = local_candidates[0]
            chosen_row = fallback["row"]
            confidence = int(round(min(fallback.get("score", 70.0), 100.0)))
            explanation = "local fuzzy match"
        else:
            results.append({
                "品名": it,
                "階": "",
                "通路": "",
                "カテゴリ(推定)": "(未判定)",
                "score": 0,
                "explanation": "not found",
            })
            detail_records.append(detail_record)
            continue

        if chosen_row is None:
            results.append({
                "品名": it,
                "階": "",
                "通路": "",
                "カテゴリ(推定)": "(未判定)",
                "score": 0,
                "explanation": "not found",
            })
        else:
            results.append({
                "品名": it,
                "階": chosen_row.floor,
                "通路": chosen_row.aisle,
                "カテゴリ(推定)": chosen_row.category,
                "score": int(max(0, min(confidence, 100))),
                "explanation": explanation,
            })

        detail_records.append(detail_record)

    df = pd.DataFrame(results)
    if not df.empty:
        def _as_key(x):
            if not x:
                return (10**9, 10**9, "")
            s = str(x)
            m = re.match(r"^(\d+)([A-Za-z�`-�ya-z��-��]*)?$", s)
            if not m:
                return (10**9, 10**9, s)
            n = int(m.group(1))
            suf = (m.group(2) or "").upper()
            order = {"A": 0, "B": 1, "C": 2, "": 3}.get(suf if suf.isalpha() else "", 99)
            return (n, order, s)

        df["_floor_ord"] = df["階"].map(lambda f: 0 if f == "1F" else (1 if f == "2F" else 2))
        df["_a_n"] = df["通路"].map(lambda x: _as_key(x)[0] if x else 10**9)
        df["_a_s"] = df["通路"].map(lambda x: _as_key(x)[1] if x else 10**9)
        df_sorted = df.sort_values(by=["_floor_ord", "_a_n", "_a_s", "通路"], ascending=[True, True, True, True]).drop(columns=["_floor_ord", "_a_n", "_a_s"])
        df_sorted.insert(0, "No.", range(1, len(df_sorted) + 1))
    else:
        df_sorted = df

    return df_sorted, detail_records
# ========== 画像（パス探索/合成/ハイライト） ==========
APP_DIR = Path(__file__).resolve().parent
CANDIDATE_ASSET_DIRS = [APP_DIR / "素材", APP_DIR / "assets", APP_DIR / "static"]

def _first_existing_path(*parts: str) -> Optional[str]:
    for root in CANDIDATE_ASSET_DIRS:
        p = root.joinpath(*parts)
        if p.exists():
            return str(p)
    return None

BASE_1F_PATH = _first_existing_path("OK1階.png")
BASE_2F_PATH = _first_existing_path("OK2階.png")

if BASE_1F_PATH is None:
    old = Path(r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\OK1階.png")
    if old.exists():
        BASE_1F_PATH = str(old)
if BASE_2F_PATH is None:
    old = Path(r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\OK2階.png")
    if old.exists():
        BASE_2F_PATH = str(old)

HIGHLIGHT_DIR_1F: Optional[str] = None
HIGHLIGHT_DIR_2F: Optional[str] = None
for root in CANDIDATE_ASSET_DIRS:
    d1 = root / "1階ハイライト"
    d2 = root / "2階ハイライト"
    if HIGHLIGHT_DIR_1F is None and d1.is_dir():
        HIGHLIGHT_DIR_1F = str(d1)
    if HIGHLIGHT_DIR_2F is None and d2.is_dir():
        HIGHLIGHT_DIR_2F = str(d2)

if HIGHLIGHT_DIR_1F is None:
    old = Path(r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\1階ハイライト")
    if old.is_dir():
        HIGHLIGHT_DIR_1F = str(old)
if HIGHLIGHT_DIR_2F is None:
    old = Path(r"C:\Users\tenne\OneDrive\Documents\■テンポラリー置場\OK_app\素材\2階ハイライト")
    if old.is_dir():
        HIGHLIGHT_DIR_2F = str(old)

try:
    from PIL.Image import Resampling  # Pillow >=10
    RESAMPLE_LANCZOS = Resampling.LANCZOS
except Exception:
    def _pick_resample_lanczos():
        from PIL import Image as _Image
        for name in ("LANCZOS", "ANTIALIAS"):
            v = getattr(_Image, name, None)
            if v is not None:
                return v
        return 1
    RESAMPLE_LANCZOS = _pick_resample_lanczos()

def pillow_overlay_image(base_rgba: Image.Image, ov_rgba: Image.Image, x: int = 0, y: int = 0, scale: float = 1.0, alpha: float = 1.0) -> Image.Image:
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

def _safe_open_rgba(path: Optional[str]) -> Optional[Image.Image]:
    try:
        if path and os.path.exists(path) and str(path).lower().endswith(".png"):
            return Image.open(path).convert("RGBA")
    except Exception:
        return None
    return None

def _normalize_aisle_token(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip()
    m = re.match(r"^0*(\d+)\s*([A-Za-z])?$", s)
    if not m:
        return s
    num = m.group(1)
    suf = (m.group(2) or "").upper()
    return f"{num}{suf}"

def _find_highlight_pngs_for_aisle(hl_dir: Optional[str], aisle: str) -> List[str]:
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

def compute_active_aisles(df_sorted: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    active: Dict[str, List[str]] = {"1F": [], "2F": []}
    if df_sorted is None or df_sorted.empty:
        return active
    valid = df_sorted[(df_sorted["通路"].notna()) & (df_sorted["通路"].astype(str).str.strip() != "")]
    if valid.empty:
        return active
    gb = valid.groupby(["階", "通路"], dropna=True)
    for keys, g in gb:
        floor_key, aisle_key = str(keys[0]), str(keys[1])
        nos = g["No."].tolist()
        all_checked = all(bool(st.session_state.get(f"done_{no}", False)) for no in nos)
        if not all_checked:
            active.setdefault(floor_key, []).append(aisle_key)
    for fl in ("1F", "2F"):
        uniq = sorted(set(active.get(fl, [])), key=lambda a: (aisle_sort_key(a)[0], aisle_sort_key(a)[1], str(a)))
        active[fl] = uniq
    return active

# ====== ★ここを刷新：縦並びで安定描画 ======
def render_floor_maps_with_auto_overlay(
    results_df: Optional[pd.DataFrame],
    force_active_aisles: Dict[str, List[str]],
    floor_filter: str,
) -> None:
    show_1f = floor_filter in ("全部", "1F")
    show_2f = floor_filter in ("全部", "2F")

    # 「全部」のときは上下に縦並びで固定化し、レイアウトの乱れを抑える
    def _render_one(floor_label: str) -> None:
        if floor_label == "1F":
            base = _safe_open_rgba(BASE_1F_PATH)
            if base is None:
                st.info("1F ベース画像が見つかりません。『素材/OK1階.png』を確認してください。")
                return
            aisles = force_active_aisles.get("1F", [])
            merged = base
            if aisles and HIGHLIGHT_DIR_1F:
                for a in aisles:
                    for png_path in _find_highlight_pngs_for_aisle(HIGHLIGHT_DIR_1F, a):
                        ov = _safe_open_rgba(png_path)
                        if ov is not None:
                            merged = pillow_overlay_image(merged, ov, 0, 0, 1.0, ALPHA_OVERLAY_1F)
            st.image(np.array(merged), caption="1F レイアウト", use_container_width=True)

        else:  # "2F"
            base = _safe_open_rgba(BASE_2F_PATH)
            if base is None:
                st.info("2F ベース画像が見つかりません。『素材/OK2階.png』を確認してください。")
                return
            aisles = force_active_aisles.get("2F", [])
            merged = base
            if aisles and HIGHLIGHT_DIR_2F:
                for a in aisles:
                    for png_path in _find_highlight_pngs_for_aisle(HIGHLIGHT_DIR_2F, a):
                        ov = _safe_open_rgba(png_path)
                        if ov is not None:
                            merged = pillow_overlay_image(merged, ov, 0, 0, 1.0, ALPHA_OVERLAY_2F)
            st.image(np.array(merged), caption="2F レイアウト", use_container_width=True)

    if show_1f and show_2f:
        # 縦並び（上下）
        with st.container():
            _render_one("1F")
        with st.container():
            _render_one("2F")
    elif show_1f:
        with st.container():
            _render_one("1F")
    elif show_2f:
        with st.container():
            _render_one("2F")
