from __future__ import annotations

from typing import Optional, List, Dict, cast

import pandas as pd
import streamlit as st

from constants import SAMPLE_INPUT, CHECKLIST_COLUMNS
from functions import (
    classify_items,
    compute_active_aisles,
    render_floor_maps_with_auto_overlay,
)
from diagnostics import sidebar_diagnostics

# ===== ページ設定 & CSS =====
st.set_page_config(page_title="Shopping Assistant", layout="wide")
st.markdown(
    """
    <style>
    :root { --checkbox-nudge: -0.75em; }
    .rowcell { height: 36px; display: flex; align-items: center; }
    div[data-testid="stCheckbox"] { height: 36px; display: flex; align-items: center; transform: translateY(var(--checkbox-nudge)); }
    @media (max-width: 640px) {
      .rowcell { height: 40px; }
      .stButton>button { padding: 0.8rem 1.0rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Shopping Assistant")

# ===== 先に「例を貼る」「クリア」の遷移フラグを処理（ウィジェット生成前に実行） =====
if st.session_state.get("__paste_sample__", False):
    # サンプルをテキストエリアへ注入（ウィジェット作成前）
    st.session_state["items_text_area"] = SAMPLE_INPUT
    st.session_state["__paste_sample__"] = False
    st.session_state["__suppress_next_on_change"] = True
    st.session_state["__last_fill_mode"] = "sample"

if st.session_state.get("__clear_pending__", False):
    # チェックボックス全解除 & 結果クリア
    for k in list(st.session_state.keys()):
        if str(k).startswith("done_"):
            del st.session_state[k]
    st.session_state["__last_results__"] = None
    st.session_state["__last_details__"] = None

    # ★手入力/サンプルに関わらず常に入力欄を空にする
    st.session_state["items_text_area"] = ""
    st.session_state["__last_fill_mode"] = None
    st.session_state["__suppress_next_on_change"] = False
    st.session_state["__clear_pending__"] = False

# ===== サイドバー（上：買い物リスト / 下：AI設定） =====
with st.sidebar:
    st.header("買い物リスト")

    def _on_text_change():
        # サンプル注入直後の on_change を一度だけ無効化
        if st.session_state.get("__suppress_next_on_change"):
            st.session_state["__suppress_next_on_change"] = False
            return
        st.session_state["__last_fill_mode"] = "manual"

    items_text = st.text_area(
        "1行に1品を入力",
        key="items_text_area",
        height=320,  # 入力スペース広め
        placeholder="例）\nガムテープ\nリンゴジュース\n…",
        on_change=_on_text_change,
    )

    # 検索ボタン（フル幅）
    run = st.button("検索", type="primary", use_container_width=True)

    # ★ クリアボタンを検索の直下に半幅で配置（サイズは以前のまま）
    clear_col, _spacer = st.columns(2)
    with clear_col:
        if st.button("クリア", use_container_width=True):
            st.session_state["__clear_pending__"] = True
            st.rerun()

    st.divider()
    st.subheader("表示")
    floor_filter = st.radio("フロア表示", ["全部", "1F", "2F"], index=0, horizontal=True)

    with st.expander("AI設定（必要時のみ）", expanded=False):
        mode = st.radio(
            "判定モード",
            ["AI (LLM + Embeddings)", "AI (Embeddings only)", "Local (RapidFuzz)"],
            index=0,
        )
        embed_model = st.selectbox(
            "Embeddings モデル", ["text-embedding-3-small", "text-embedding-3-large"], index=0
        )
        chat_model = st.selectbox("LLM モデル", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
        batch_llm = st.checkbox("LLM まとめて1回判定（高速）", value=True)
        topk = st.slider("候補K（Embeddingsで絞り込み）", 5, 20, 10)
        embed_conf_threshold = st.slider("埋め込みだけで確定（cos類似の閾値）", 0.80, 0.98, 0.92, 0.01)
        use_external = st.checkbox("無料API（OFF/OBF/OPF）候補補強", value=True)

        # 例を貼る（AI設定の一番下・半幅）
        col1, col2 = st.columns(2)
        with col1:
            if st.button("例を貼る", use_container_width=True):
                st.session_state["__paste_sample__"] = True
                st.rerun()
        # col2 は空（半幅のまま）

# ===== 診断（サイドバー最下部） =====
sidebar_diagnostics()

# ===== 実行 =====
if run:
    items = [s for s in (items_text or "").splitlines() if s.strip()]
    if not items:
        st.warning("品名を入力してください。")
    else:
        with st.spinner("分類中…"):
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

# ===== 本文：左右2カラムレイアウト =====
left, right = st.columns([0.54, 0.46])

# 左：チェックリスト
with left:
    results_df = cast(Optional[pd.DataFrame], st.session_state.get("__last_results__"))
    if results_df is not None and not results_df.empty:
        st.subheader("チェックリスト")

        view_df = results_df[CHECKLIST_COLUMNS].copy()
        hcols = st.columns([0.12, 0.12, 0.48, 0.12, 0.16])
        for col, name in zip(hcols, ["✓", "No.", "品名", "階", "通路"]):
            col.markdown(f"<div class='rowcell'><b>{name}</b></div>", unsafe_allow_html=True)

        prev_floor: Optional[str] = None
        for _, row in view_df.iterrows():
            if prev_floor is not None and prev_floor != row["階"]:
                st.markdown(
                    "<div style='border-top:2px dashed #c7c7c7; margin:6px 0;'></div>",
                    unsafe_allow_html=True,
                )
            prev_floor = row["階"]  # type: ignore[index]

            no = int(row["No."])  # type: ignore[index]
            key = f"done_{no}"
            checked = bool(st.session_state.get(key, False))

            cols = st.columns([0.12, 0.12, 0.48, 0.12, 0.16])
            cols[0].checkbox(" ", value=checked, key=key, label_visibility="collapsed")

            def _grey(text: str) -> str:
                style = "color:#9aa0a6; opacity:0.65" if st.session_state[key] else ""
                return f"<div class='rowcell'><span style='{style}'>{text}</span></div>"

            cols[1].markdown(_grey(str(row["No."])), unsafe_allow_html=True)
            cols[2].markdown(_grey(str(row["品名"])), unsafe_allow_html=True)
            cols[3].markdown(_grey(str(row["階"])), unsafe_allow_html=True)
            cols[4].markdown(_grey(str(row["通路"])), unsafe_allow_html=True)
    else:
        st.info("「検索」を押すと結果が表示されます。")

# 右：フロアマップ（「全部」のときは 1F→2F を縦に積む）
with right:
    st.subheader("フロアマップ")
    results_df = cast(Optional[pd.DataFrame], st.session_state.get("__last_results__"))
    active_aisles: Dict[str, List[str]] = compute_active_aisles(results_df)

    if floor_filter == "全部":
        render_floor_maps_with_auto_overlay(results_df, active_aisles, "1F")
        render_floor_maps_with_auto_overlay(results_df, active_aisles, "2F")
    else:
        render_floor_maps_with_auto_overlay(results_df, active_aisles, floor_filter)
