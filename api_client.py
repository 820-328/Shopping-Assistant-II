from __future__ import annotations

import os
from typing import Dict, List, Optional, Any, TypedDict

import requests

# ===== 外部API（OpenFood/Beauty/Products Facts） =====

HTTP_TIMEOUT = 6.0  # 秒: 外部API呼び出しのタイムアウト
REQUEST_HEADERS = {
    "User-Agent": "ShoppingAssistant/1.0 (+contact: you@example.com)"
}

OFF_URL = "https://world.openfoodfacts.org/api/v2/search"
OBF_URL = "https://world.openbeautyfacts.org/api/v2/search"
OPF_URL = "https://world.openproductsfacts.org/api/v2/search"


def _strip_lang(tag: str) -> str:
    return tag.split(":", 1)[1] if ":" in tag else tag


def _fetch_json(url: str, params: Dict) -> Dict:
    """GET→JSON 取得（失敗時は {} を返す）"""
    try:
        r = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def external_hints(q: str) -> Dict:
    """
    既存の「カテゴリタグ候補」関数。
    tags: 統合カテゴリタグ上位（去重）
    sources: 各APIの元タグ
    """
    off = _fetch_json(OFF_URL, {"search_terms": q, "page_size": 12, "fields": "product_name,categories_tags"})
    obf = _fetch_json(OBF_URL, {"search_terms": q, "page_size": 12, "fields": "product_name,categories_tags"})
    opf = _fetch_json(OPF_URL, {"search_terms": q, "page_size": 12, "fields": "product_name,categories_tags"})

    def _collect_tags(js):
        s = set()
        for p in (js.get("products") or []):
            for t in (p.get("categories_tags") or []):
                s.add(_strip_lang(t))
        return list(s)

    off_tags = _collect_tags(off)
    obf_tags = _collect_tags(obf)
    opf_tags = _collect_tags(opf)

    merged = []
    seen = set()
    for lst in (off_tags, obf_tags, opf_tags):
        merged.extend(lst or [])

    out: List[str] = []
    for t in merged:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 20:
            break

    return {"tags": out, "sources": {"off": off_tags[:20], "obf": obf_tags[:20], "opf": opf_tags[:20]}}


# 診断用の厳密な型（Pylance対策）
class SourceResult(TypedDict):
    count: int
    names: List[str]
    tags: List[str]


class ProbeResult(TypedDict, total=False):
    ok: bool
    off: SourceResult
    obf: SourceResult
    opf: SourceResult


def probe_external_api(q: str, page_size: int = 8) -> ProbeResult:
    """
    診断用：商品名（product_name）とカテゴリタグの取得テスト。
    返却フォーマット:
    {
      "ok": bool,
      "off": {"count": 12, "names": [...], "tags": [...]},
      "obf": {...},
      "opf": {...}
    }
    """
    params = {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"}

    off = _fetch_json(OFF_URL, params)
    obf = _fetch_json(OBF_URL, params)
    opf = _fetch_json(OPF_URL, params)

    def _names(js: Dict) -> List[str]:
        return [p.get("product_name") for p in (js.get("products") or []) if p.get("product_name")]

    def _tags(js: Dict) -> List[str]:
        s = set()
        for p in (js.get("products") or []):
            for t in (p.get("categories_tags") or []):
                s.add(_strip_lang(t))
        return list(s)

    res: ProbeResult = {
        "off": {"count": len(off.get("products") or []), "names": _names(off), "tags": _tags(off)},
        "obf": {"count": len(obf.get("products") or []), "names": _names(obf), "tags": _tags(obf)},
        "opf": {"count": len(opf.get("products") or []), "names": _names(opf), "tags": _tags(opf)},
    }
    res["ok"] = any((res["off"]["count"], res["obf"]["count"], res["opf"]["count"]))
    return res


# ===== OpenAI クライアント =====

# アプリ全体で参照する「処理タイムアウト（秒）」推奨値
TIMEOUT_CHAT = 120.0
TIMEOUT_EMBEDDINGS = 45.0


def _get_api_key() -> Optional[str]:
    # まず環境変数を優先
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    # st.secrets は診断側での表示に限定し、ここでは触らない
    return None


def get_openai_client():
    """
    OpenAI 1.x クライアントを返す。キーが無い/失敗時は None。
    """
    try:
        from openai import OpenAI  # import 失敗時は None を返す
    except Exception:
        return None

    key = _get_api_key()
    if not key:
        return None

    try:
        # 1.x の標準初期化（proxies 等は渡さない）
        return OpenAI(api_key=key)
    except Exception:
        return None
