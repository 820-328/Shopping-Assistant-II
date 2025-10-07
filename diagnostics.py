from __future__ import annotations

import os
import time
import json
from typing import Any, Dict

import streamlit as st
import pandas as pd

# 他モジュールへの依存は最小限にし、ここでのみ import します
from api_client import get_openai_client, probe_external_api

__all__ = ["sidebar_diagnostics"]

# ===== バージョン表示用 =====
try:
    import streamlit as _st
    STREAMLIT_VER = _st.__version__
except Exception:
    STREAMLIT_VER = "(unknown)"

try:
    import openai as _openai_pkg
    OPENAI_VER = getattr(_openai_pkg, "__version__", "(unknown)")
except Exception:
    OPENAI_VER = "(unknown)"


def _one_time_healthcheck(client) -> str:
    """OpenAI クライアントの疎通を 1 回だけ実行して文字列を返す。"""
    if client is None:
        return "❌ OpenAI: クライアント未初期化（APIキー未検出？）"
    try:
        t0 = time.time()
        _ = client.models.list()
        ms = int((time.time() - t0) * 1000)
        return f"✅ OpenAI: 認証OK（{ms} ms）"
    except Exception as e:
        return f"❌ OpenAI: 失敗 — {e!r}"


@st.cache_data(show_spinner=False)
def _cached_health(_dummy: int = 0) -> str:
    """client は非ハッシュ（プリミティブだけをハッシュ対象にする）"""
    client = get_openai_client()
    return _one_time_healthcheck(client)


def _render_api_probe(res: Dict[str, Any]) -> None:
    """APIテスト結果を、読みやすい表＋任意の Raw JSON で表示。"""
    status = "✅ 成功（いずれかでヒット）" if res.get("ok") else "❌ 0件（全てヒットなし）"
    st.write(status)

    rows = []
    for src in ("off", "obf", "opf"):
        d = res.get(src, {})
        if isinstance(d, dict):
            cnt = int(d.get("count", 0))
            names = ", ".join([n for n in (d.get("names") or [])[:3] if isinstance(n, str)])
            tags = ", ".join([t for t in (d.get("tags") or [])[:5] if isinstance(t, str)])
        else:
            cnt, names, tags = 0, "", ""
        rows.append({"ソース": src.upper(), "件数": cnt, "名前例": names, "タグ例": tags})

    df = pd.DataFrame(rows, columns=["ソース", "件数", "名前例", "タグ例"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    # 入れ子 Expander は禁止（Streamlit 制約）。チェックボックスで Raw JSON を開閉。
    show_raw = st.checkbox("Raw JSON（詳細）を表示", value=False, key="__probe_show_raw__")
    if show_raw:
        st.code(json.dumps(res, ensure_ascii=False, indent=2), language="json")


def _mask_key(k: str | None) -> str:
    if not k:
        return "（未設定）"
    # 先頭4文字だけ表示
    return f"{k[:4]}…"


def sidebar_diagnostics() -> None:
    """サイドバーに診断 UI を描画。"""
    with st.sidebar:
        st.divider()
        st.subheader("診断")
        st.caption(f"Streamlit {STREAMLIT_VER} / openai {OPENAI_VER}")

        # APIキーの存在表示（環境変数優先。secrets は任意）
        key_env = os.getenv("OPENAI_API_KEY")
        key_sec = None
        try:
            key_sec = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            pass

        key_display = key_env or (str(key_sec) if key_sec else "")
        if key_display:
            st.caption(f"APIキー検出: ✅ ({_mask_key(key_display)})")
        else:
            st.caption("APIキー検出: ❌（.env に OPENAI_API_KEY=sk-... を設定）")

        # ヘルスチェック（キャッシュ）
        st.caption(_cached_health(0))

        # ====== APIテスト：商品名ヒント ======
        with st.expander("APIテスト（商品名ヒント）", expanded=False):
            default_q = st.session_state.get("__probe_q__", "はちみつ")
            q: str = st.text_input("検索語（例：はちみつ）", value=str(default_q), key="__probe_q__")
            run = st.button("実行", use_container_width=True, key="__probe_run__")

            if run:
                with st.spinner("問い合わせ中…"):
                    res = probe_external_api(q or "")
                st.session_state["__probe_result__"] = res

            res_any = st.session_state.get("__probe_result__", None)
            if isinstance(res_any, dict):
                _render_api_probe(res_any)
            else:
                st.caption("未実行です。検索語を入力して「実行」を押してください。")
