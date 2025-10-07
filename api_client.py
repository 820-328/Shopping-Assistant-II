# api_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# .env を可能ならロード（python-dotenv が無ければスキップ）
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


# ─────────────────────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    """
    環境変数（.env を含む）から API キーを取得。
    ※ st.secrets は使わない（「No secrets found」警告を出さないため）
    """
    return os.getenv("OPENAI_API_KEY") or None


@st.cache_resource(show_spinner=False)
def get_openai_client(cache_buster: str = "") -> Optional[OpenAI]:
    """
    OpenAI クライアントを返す。キーが無ければ None。
    SDK 1.109.1 用。cache_buster はキャッシュ無効化用のダミー引数。
    """
    key = get_api_key()
    if not key:
        return None

    # 内部で参照される場合に備えて環境変数にも反映
    os.environ.setdefault("OPENAI_API_KEY", key)

    try:
        client = OpenAI(api_key=key, timeout=10.0, max_retries=1)
        return client
    except Exception as e:
        # 診断側で参照できるよう session_state に格納（文字列で固定）
        st.session_state["__api_client_error__"] = f"OpenAI init failed: {e!r}"
        return None


# ─────────────────────────────────────────────────────────────
# External product-name hints (OFF / OBF / OPF)
# ─────────────────────────────────────────────────────────────

HEADERS: Dict[str, str] = {
    "User-Agent": "ShoppingAssistant/1.0 (+contact: you@example.com)"
}


def _fetch_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _strip_lang(tag: str) -> str:
    # 'en:beverages' → 'beverages'
    return tag.split(":", 1)[1] if ":" in tag else tag


def _off_search(q: str, page_size: int = 12) -> Dict[str, Any]:
    js = _fetch_json(
        "https://world.openfoodfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    names: List[str] = []
    tags: List[str] = []
    for p in (js.get("products") or []):
        pname = p.get("product_name")
        if isinstance(pname, str) and pname:
            names.append(pname)
        for t in (p.get("categories_tags") or []):
            if isinstance(t, str):
                tags.append(_strip_lang(t))
    return {"count": len(js.get("products") or []), "names": names, "tags": tags}


def _obf_search(q: str, page_size: int = 12) -> Dict[str, Any]:
    js = _fetch_json(
        "https://world.openbeautyfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    names: List[str] = []
    tags: List[str] = []
    for p in (js.get("products") or []):
        pname = p.get("product_name")
        if isinstance(pname, str) and pname:
            names.append(pname)
        for t in (p.get("categories_tags") or []):
            if isinstance(t, str):
                tags.append(_strip_lang(t))
    return {"count": len(js.get("products") or []), "names": names, "tags": tags}


def _opf_search(q: str, page_size: int = 12) -> Dict[str, Any]:
    js = _fetch_json(
        "https://world.openproductsfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    names: List[str] = []
    tags: List[str] = []
    for p in (js.get("products") or []):
        pname = p.get("product_name")
        if isinstance(pname, str) and pname:
            names.append(pname)
        for t in (p.get("categories_tags") or []):
            if isinstance(t, str):
                tags.append(_strip_lang(t))
    return {"count": len(js.get("products") or []), "names": names, "tags": tags}


@st.cache_data(show_spinner=False, ttl=3600)
def probe_external_api(q: str) -> Dict[str, Any]:
    """
    診断用：OFF/OBF/OPF を横断して件数・サンプル名・タグを返す。
    """
    query = (q or "").strip()
    if not query:
        return {"ok": False, "off": {}, "obf": {}, "opf": {}}

    off = _off_search(query)
    obf = _obf_search(query)
    opf = _opf_search(query)
    ok = any([off.get("count", 0), obf.get("count", 0), opf.get("count", 0)])
    return {"ok": bool(ok), "off": off, "obf": obf, "opf": opf}


# 互換API：旧コードが参照する external_hints を提供
@st.cache_data(show_spinner=False, ttl=3600)
def external_hints(q: str, max_tags: int = 20) -> Dict[str, Any]:
    """
    旧 ai/functions が期待していた形式:
      { "tags": [...], "sources": {"off":[...],"obf":[...],"opf":[...]} }
    """
    res = probe_external_api(q or "")

    def _top_tags(d: Any) -> List[str]:
        if isinstance(d, dict):
            raw = d.get("tags") or []
            return [t for t in raw if isinstance(t, str)]
        return []

    off_tags = _top_tags(res.get("off"))
    obf_tags = _top_tags(res.get("obf"))
    opf_tags = _top_tags(res.get("opf"))

    merged: List[str] = []
    for chunk in (off_tags, obf_tags, opf_tags):
        merged.extend(chunk)

    # 重複排除して max_tags までに整形
    seen: set[str] = set()
    uniq: List[str] = []
    for t in merged:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_tags:
            break

    return {
        "tags": uniq,
        "sources": {
            "off": off_tags[:max_tags],
            "obf": obf_tags[:max_tags],
            "opf": opf_tags[:max_tags],
        },
    }
