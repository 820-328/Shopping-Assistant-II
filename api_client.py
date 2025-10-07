from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from openai import OpenAI

# ── .env を可能ならロード（python-dotenv が無ければスキップ）
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# ============================================
# OpenAI Client
# ============================================

def _get_api_key() -> Optional[str]:
    """
    ローカルでは .env / 環境変数のみを見る。
    （st.secrets を参照しないことで「No secrets found」警告を回避）
    """
    key = os.getenv("OPENAI_API_KEY")
    return key if key else None


@st.cache_resource(show_spinner=False)
def get_openai_client() -> Optional[OpenAI]:
    """
    診断や推論で使う OpenAI クライアント。
    1.109.1 系 SDK に合わせ、api_key のみ明示（proxies等は渡さない）。
    """
    key = _get_api_key()
    if not key:
        return None

    # 念のため環境変数にも反映（SDK内部参照のケースに備える）
    os.environ.setdefault("OPENAI_API_KEY", key)

    try:
        client = OpenAI(api_key=key, timeout=10.0, max_retries=1)
        return client
    except Exception as e:
        st.session_state["__api_client_error__"] = f"OpenAI init failed: {e!r}"
        return None

# ============================================
# External product-name hints (OFF / OBF / OPF)
# ============================================

HEADERS = {"User-Agent": "ShoppingAssistant/1.0 (+contact: you@example.com)"}

def _fetch_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def _strip_lang(tag: str) -> str:
    return tag.split(":", 1)[1] if ":" in tag else tag

def _off_search(q: str, page_size: int = 12) -> Dict[str, Any]:
    js = _fetch_json(
        "https://world.openfoodfacts.org/api/v2/search",
        {"search_terms": q, "page_size": page_size, "fields": "product_name,categories_tags"},
    )
    names: List[str] = []
    tags: List[str] = []
    for p in (js.get("products") or []):
        if p.get("product_name"):
            names.append(p["product_name"])
        for t in (p.get("categories_tags") or []):
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
        if p.get("product_name"):
            names.append(p["product_name"])
        for t in (p.get("categories_tags") or []):
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
        if p.get("product_name"):
            names.append(p["product_name"])
        for t in (p.get("categories_tags") or []):
            tags.append(_strip_lang(t))
    return {"count": len(js.get("products") or []), "names": names, "tags": tags}

@st.cache_data(show_spinner=False, ttl=3600)
def probe_external_api(q: str) -> Dict[str, Any]:
    """診断用：OFF/OBF/OPF を横断して件数・サンプル名・タグを返す。"""
    q = (q or "").strip()
    if not q:
        return {"ok": False, "off": {}, "obf": {}, "opf": {}}

    off = _off_search(q)
    obf = _obf_search(q)
    opf = _opf_search(q)
    ok = any([off.get("count", 0), obf.get("count", 0), opf.get("count", 0)])
    return {"ok": bool(ok), "off": off, "obf": obf, "opf": opf}

# ---- 互換API：旧コードが参照する external_hints を提供
@st.cache_data(show_spinner=False, ttl=3600)
def external_hints(q: str, max_tags: int = 20) -> Dict[str, Any]:
    """
    旧 ai/functions が期待していた形式:
      { "tags": [...], "sources": {"off":[...],"obf":[...],"opf":[...]} }
    """
    res = probe_external_api(q or "")

    def _top_tags(d: Any) -> List[str]:
        if isinstance(d, dict):
            t = d.get("tags") or []
            return [x for x in t if isinstance(x, str)]
        return []

    off_tags = _top_tags(res.get("off"))
    obf_tags = _top_tags(res.get("obf"))
    opf_tags = _top_tags(res.get("opf"))

    merged: List[str] = []
    for chunk in (off_tags, obf_tags, opf_tags):
        merged.extend(chunk)

    seen = set()
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
