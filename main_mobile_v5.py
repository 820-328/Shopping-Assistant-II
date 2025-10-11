"""
Shopping Assistant mobile-friendly UI (v5.08).
スマートフォンでも見やすいレイアウト。AI 設定とサンプル操作ガイドはサイドバー下部へ統一。
"""

from __future__ import annotations

from typing import Dict, List, Optional, cast

import html
import pandas as pd
import streamlit as st

from constants import CHECKLIST_COLUMNS, SAMPLE_INPUT
from diagnostics import sidebar_diagnostics
from functions import (
    classify_items,
    compute_active_aisles,
    render_floor_maps_with_auto_overlay,
)


# ===== ページ設定 / CSS ====================================================
st.set_page_config(
    page_title="Shopping Assistant（モバイル）",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MAX_CONTENT_PX = 540
st.markdown(
    f"""
    <style>
    html, body {{
      overscroll-behavior-y: none;
      background: #f4f5f7;
    }}
    :root {{ color-scheme: light; }}
    .block-container {{
      max-width: min(1200px, 95vw) !important;
      padding: 0.75rem 1.2rem 5rem;
    }}
    .mobile-card {{
      border-radius: 14px;
      padding: 0.65rem 0.75rem;
      background: white;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
      transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
    }}
    .mobile-card.done {{
      opacity: 0.48;
      background: #f8fafc;
    }}
    .mobile-title {{
      font-size: 1rem;
      font-weight: 600;
      line-height: 1.35;
      word-break: break-word;
    }}
    .mobile-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.4rem;
      font-size: 0.78rem;
      color: #64748b;
    }}
    .mobile-meta.detail-line {{
      display: block;
      gap: 0.25rem;
      font-size: 0.74rem;
      color: #475569;
    }}
    .mobile-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
      background: #eef2ff;
      color: #4338ca;
      font-weight: 600;
    }}
    .action-row button {{ width: 100%; }}
    @media (max-width: 640px) {{
      .block-container {{
        max-width: {MAX_CONTENT_PX}px !important;
        padding-left: 0.65rem;
        padding-right: 0.65rem;
      }}
      .mobile-title {{ font-size: clamp(0.95rem, 1.8vw + 0.8rem, 1.05rem); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 追加のテーマ上書き（f文字列ではないブロックで安全に適用）
st.markdown(
    """
<style>
:root {
  --sa-orange: #F97316;   /* オレンジ */
  --sa-vermilion: #E34234;/* 朱色 */
  --primary-color: var(--sa-orange);
  --secondary-background-color: #FFF6EF;
  --text-color: #1F2937;
  --background-color: #F7F5F2;
}

/* アプリ背景とテキスト色 */
[data-testid="stAppViewContainer"] {
  background: var(--background-color) !important;
  color: var(--text-color) !important;
}

/* サイドバーと出し入れボタン（＞）を強調 */
[data-testid="stSidebar"] {
  border-right: 3px solid var(--sa-vermilion);
  background: #FFF6EF;
}
[data-testid="collapsedControl"] {
  position: fixed;
  left: 0;
  top: 64px;
  z-index: 1000;
}
[data-testid="collapsedControl"] button {
  background: var(--sa-orange) !important;
  color: #fff !important;
  border: 2px solid var(--sa-vermilion) !important;
  width: 48px; height: 48px;
  border-radius: 0 12px 12px 0;
  box-shadow: 0 6px 18px rgba(0,0,0,0.22);
}
[data-testid="collapsedControl"] svg {
  width: 22px; height: 22px;
  fill: #fff !important;
}

/* 検索ボタンなどの色 */
.stButton > button {
  background: var(--sa-orange) !important;
  border-color: var(--sa-vermilion) !important;
  color: #fff !important;
}
.stButton > button:hover { filter: brightness(0.95); }

/* スライダー色（つまみとアクティブトラック） */
[data-testid="stSlider"] [role="slider"] {
  background: var(--sa-vermilion) !important;
  border: 2px solid var(--sa-vermilion) !important;
  box-shadow: 0 0 0 4px rgba(227,66,52,0.18) !important;
}
/* アクティブトラックの色調整（バージョン間の差を吸収するため汎用セレクタ） */
[data-testid="stSlider"] div[aria-hidden="true"] > div {
  background: var(--sa-orange) !important;
}

/* ピルの配色をシンボルカラーへ */
.mobile-pill {
  background: #FFF1E6;
  color: var(--sa-vermilion);
  font-weight: 700;
}

/* モバイルで上部が切れないよう微調整＆文字サイズを少し下げる */
@media (max-width: 640px) {
  .block-container { padding-top: calc(0.5rem + env(safe-area-inset-top)); }
  .mobile-title { font-size: 0.95rem; line-height: 1.3; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<h1 class="sa-title">Shopping Assistant <span class="sa-title-sub">（モバイル）</span></h1>
""",
    unsafe_allow_html=True,
)
st.caption("買い物リストを貼り付けて『検索』を押すと、フロアと通路を自動で提案します。")


# ===== セッションステート制御 =============================================
if st.session_state.get("__paste_sample__", False):
    st.session_state["items_text_area"] = SAMPLE_INPUT
    st.session_state["__paste_sample__"] = False
    st.session_state["__suppress_next_on_change"] = True
    st.session_state["__last_fill_mode"] = "sample"

if st.session_state.get("__clear_pending__", False):
    for key in list(st.session_state.keys()):
        if str(key).startswith("done_"):
            del st.session_state[key]
    st.session_state["__last_results__"] = None
    st.session_state["__last_details__"] = None
    st.session_state["items_text_area"] = ""
    st.session_state["__last_fill_mode"] = None
    st.session_state["__suppress_next_on_change"] = False
    st.session_state["__clear_pending__"] = False


def _on_text_change() -> None:
    if st.session_state.get("__suppress_next_on_change"):
        st.session_state["__suppress_next_on_change"] = False
        return
    st.session_state["__last_fill_mode"] = "manual"


# ===== 入力フォーム ========================================================
items_text = st.text_area(
    "買い物リスト（1行1アイテム）",
    key="items_text_area",
    height=260,
    placeholder="例)\n牛乳\n食パン\nトマト缶\n…",
    on_change=_on_text_change,
)

action_cols = st.columns(2)
with action_cols[0]:
    run_clicked = st.button("検索", type="primary", use_container_width=True)
with action_cols[1]:
    if st.button("リセット", use_container_width=True):
        st.session_state["__clear_pending__"] = True
        st.rerun()

st.divider()

# VS Code 未定義警告の回避用デフォルト（サイドバーで上書きされます）
mode: str = "Local (RapidFuzz)"
embed_model: str = "text-embedding-3-small"
chat_model: str = "gpt-4o-mini"
batch_llm: bool = True
topk: int = 10
embed_conf_threshold: float = 0.92
use_external: bool = True


# ===== 表示設定 & プレースホルダー =====================================
st.subheader("表示設定")
floor_filter = st.radio("フロア表示", ["全体", "1F", "2F"], horizontal=True, index=0)

results_container = st.container()

st.divider()


# ===== サイドバー（下部にガイド / AI 設定 / 診断） ======================
with st.sidebar:
    st.divider()
    with st.expander("サンプル / 操作ガイド", expanded=False):
        st.markdown(
            "- クリップボードのリストを貼り付けると素早く試せます。\n"
            "- 各チェックに応じてアクティブ通路が更新されます。"
        )
        if st.button("サンプルリストを貼り付け", use_container_width=True, key="__paste_sample_btn__"):
            st.session_state["__paste_sample__"] = True
            st.rerun()

    with st.expander("AI 設定（必要な場合のみ）", expanded=False):
        mode = st.radio(
            "検索モード",
            ["AI (LLM + Embeddings)", "AI (Embeddings only)", "Local (RapidFuzz)"],
            index=2,
        )
        embed_model = st.selectbox(
            "Embeddings モデル",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
        )
        chat_model = st.selectbox(
            "LLM モデル",
            ["gpt-4o-mini", "gpt-4.1-mini"],
            index=0,
        )
        batch_llm = st.checkbox("LLM をまとめて 1 度呼び出す", value=True)
        topk = st.slider("候補数（Embeddings ショートリスト）", 5, 20, 10)
        embed_conf_threshold = st.slider("Embeddings 類似度しきい値", 0.80, 0.98, 0.92, 0.01)
        use_external = st.checkbox("外部 API（OFF/OBF/OPF）を補助利用", value=True)

sidebar_diagnostics()


# ===== 検索実行 ============================================================
if run_clicked:
    raw_items = items_text or ""
    items: List[str] = [line for line in raw_items.splitlines() if line.strip()]
    if not items:
        st.warning("アイテム名を入力してください。")
    else:
        with st.spinner("検索しています…"):
            df_sorted, details = classify_items(
                items=items,
                mode=mode,
                embed_model=embed_model,
                chat_model=chat_model,
                batch_llm=batch_llm,
                topk=topk,
                embed_conf_threshold=float(embed_conf_threshold),
                use_external=use_external,
            )
        st.session_state["__last_results__"] = df_sorted.copy()
        st.session_state["__last_details__"] = details


# ===== チェックリスト & フロアマップ（プレースホルダーに出力） ============
results_df = cast(Optional[pd.DataFrame], st.session_state.get("__last_results__"))
raw_details = st.session_state.get("__last_details__")
detail_map: Dict[str, Dict[str, object]] = {}
if isinstance(raw_details, list):
    for entry in raw_details:
        if isinstance(entry, dict) and entry.get("item") is not None:
            key = str(entry.get("item"))
            detail_map[key] = entry  # type: ignore[assignment]
        elif isinstance(entry, tuple) and len(entry) == 2:
            key = str(entry[0])
            detail_map[key] = {"item": entry[0], "local": entry[1], "ai": []}
        elif isinstance(entry, tuple) and len(entry) == 3:
            key = str(entry[0])
            detail_map[key] = {"item": entry[0], "local": entry[1], "ai": entry[2]}
name_col = CHECKLIST_COLUMNS[1]
floor_col = CHECKLIST_COLUMNS[2]
aisle_col = CHECKLIST_COLUMNS[3]

active_aisles: Dict[str, List[str]] = compute_active_aisles(results_df)
floor_title_map = {"1F": "1階", "2F": "2階"}
floors_to_show: List[str] = ["1F", "2F"] if floor_filter == "全体" else [floor_filter]

floor_records: Dict[str, List[dict]] = {key: [] for key in floor_title_map}
if results_df is not None and not results_df.empty:
    view_df = results_df[CHECKLIST_COLUMNS].copy()
    for record in view_df.to_dict("records"):
        floor_key = str(record.get(floor_col, "")).strip()
        floor_records.setdefault(floor_key, []).append(record)

with results_container:
    st.subheader("チェックリスト")
    for floor in floors_to_show:
        title = floor_title_map.get(floor, floor)
        st.markdown(f"--- {title} ---")
        list_col, map_col = st.columns([0.46, 0.54])

        rows = floor_records.get(floor, [])
        if not rows:
            with list_col:
                if results_df is None or results_df.empty:
                    st.info("まだ検索が実行されていません。")
                else:
                    st.info("該当アイテムはありません。")
        else:
            for row_dict in rows:
                no_raw = row_dict.get("No.", "")
                try:
                    no = int(no_raw)
                except Exception:
                    no = int(float(no_raw)) if isinstance(no_raw, (float, int)) else 0
                key = f"done_{no}"
                base_checked = bool(st.session_state.get(key, False))

                row_container = list_col.container()
                cb_col, info_col = row_container.columns([0.22, 0.78])
                cb_col.checkbox(" ", value=base_checked, key=key, label_visibility="collapsed")

                checked_now = bool(st.session_state.get(key, False))
                card_class = "mobile-card done" if checked_now else "mobile-card"

                item_name = str(row_dict.get(name_col, "") or "")
                detail_entry = detail_map.get(item_name)

                def _format_candidates(candidates: List[Dict[str, object]]) -> str:
                    formatted: List[str] = []
                    for cand in candidates[:3]:
                        category = html.escape(str(cand.get("category", "")))
                        floor_txt = str(cand.get("floor", "") or "")
                        aisle_txt = str(cand.get("aisle", "") or "")
                        location = ""
                        if floor_txt or aisle_txt:
                            location = f"({html.escape(floor_txt)}{html.escape(aisle_txt)})"
                        formatted.append(f"{category}{location}")
                    return " / ".join(formatted)

                extra_lines: List[str] = []
                if detail_entry:
                    local_list = detail_entry.get("local") or []
                    ai_list = detail_entry.get("ai") or []
                    if isinstance(local_list, list) and local_list:
                        extra_lines.append(f"① 一致候補: {_format_candidates(local_list)}")
                    if isinstance(ai_list, list) and ai_list:
                        extra_lines.append(f"② AI候補: {_format_candidates(ai_list)}")

                extra_html = "".join(
                    f"<div class=\"mobile-meta detail-line\">{html.escape(line)}</div>"
                    for line in extra_lines
                )

                info_col.markdown(
                    f"""
                    <div class="{card_class}">
                      <div class="mobile-title">{row_dict.get(name_col, '')}</div>
                      <div class="mobile-meta">
                        <span class="mobile-pill">No.{no}</span>
                        <span>{title}</span>
                        <span>{row_dict.get(aisle_col, '')}</span>
                      </div>
                      {extra_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with map_col:
            render_floor_maps_with_auto_overlay(results_df, active_aisles, floor)

# ----- End -----
