import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
from matplotlib import font_manager
import time
from datetime import datetime, timedelta

matplotlib.rcParams['font.family'] = ['IPAexGothic', 'Noto Sans CJK JP', 'Yu Gothic', 'sans-serif'] # または 'Yu Gothic', 'Noto Sans CJK JP' など



# --- ページ設定 ---
st.set_page_config(page_title="埼玉データ分析アプリ", page_icon="📊", layout="wide")

# --- スタイル設定 ---
st.markdown("""
<style>
body {
    background-color: #f0f4f8;
    color: #1a1a1a;
    font-family: "Helvetica Neue", sans-serif;
}
section.main > div {
    padding: 2rem;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0px 2px 12px rgba(0, 0, 0, 0.05);
}
h1, h2, h3, h4 {
    color: #0d3b66;
}
.stButton > button {
    background-color: #3e92cc !important;
    color: white !important;
    border: none;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    font-weight: bold;
    transition: 0.3s;
    font-family: inherit;
    font-size: 16px;
}
.stButton > button:hover {
    background-color: #265d88;
    color: #f1f1f1;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# --- セッション状態の初期化 ---
if "graph_shown" not in st.session_state:
    st.session_state.graph_shown = False

if "graph_button_clicked" not in st.session_state:
    st.session_state.graph_button_clicked = False

if "analyze_shown" not in st.session_state:
    st.session_state.analyze_shown = False

# --- データ読み込み ---
@st.cache_data


def load_data():
    df = pd.read_csv("saitama_data2.csv")
    df = df[df["調査年"].notna()]  # 調査年が空でないもの
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.drop(columns=[col for col in ["地域"] if col in df.columns])
    for col in df.columns:
        if col != "調査年":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# --- データ読み込み ---
df = load_data()
#valid_pairs = precompute_valid_pairs(df)
numeric_columns = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]
x_candidates = numeric_columns

# --- タイトル ---
st.title("埼玉県オープンデータ分析体験")

# --- サイドバー設定 ---
with st.sidebar:
    st.markdown("## 🎛 データ設定")

    # --- X軸の選択 ---
    prev_x = st.session_state.get("prev_x")
    x_col = st.selectbox("X軸にする項目", x_candidates, key="x")
    if prev_x is not None and prev_x != x_col:
        st.session_state.graph_shown = False
        st.session_state.analyze_shown = False
        st.session_state.graph_button_clicked = False
    st.session_state.prev_x = x_col

    # --- Y軸の選択 ---
    x_head = x_col[0]

    # 人口データ（例：Aで始まる列名）
    def is_population(col): return col.startswith("A")

    # 図書館データ（列名に"図書館"が含まれる）
    def is_library(col): return "図書館" in col

    # Y軸候補フィルタ：Xと同じ列／同じイニシャル／人口×図書館の組み合わせ → 除外
    y_candidates = []
    for col in numeric_columns:
        if col == x_col:
            continue
        if col[0] == x_head:
            continue
        if (is_population(x_col) and is_library(col)) or (is_library(x_col) and is_population(col)):
            continue
        y_candidates.append(col)

    # ✅ forループの外で選択＆状態管理
    prev_y = st.session_state.get("prev_y")
    y_col = st.selectbox("Y軸にする項目", y_candidates, key="y")
    if prev_y is not None and prev_y != y_col:
        st.session_state.graph_shown = False
        st.session_state.analyze_shown = False
        st.session_state.graph_button_clicked = False
    st.session_state.prev_y = y_col

    # --- グラフ種類の選択 ---
    previous_graph_type = st.session_state.get("previous_graph_type")
    graph_type = st.radio("表示するグラフの種類", [
        "散布図", "折れ線グラフ", "棒グラフ", "円グラフ", "ヒストグラム", "箱ひげ図"
    ], key="graph")

    if previous_graph_type is not None and previous_graph_type != graph_type:
        st.session_state.graph_shown = False
        st.session_state.analyze_shown = False
        st.session_state.graph_button_clicked = False
    st.session_state.previous_graph_type = graph_type

    if st.button("📈 グラフ化"):
        st.session_state.graph_button_clicked = True
        st.session_state.analyze_shown = False


# --- グラフ表示 ---
if st.session_state.graph_button_clicked:
    st.session_state.graph_shown = True

if st.session_state.graph_shown:
    st.markdown("## グラフ表示")

    # --- グラフの説明文 ---
    graph_explanations = {
        "散布図": "散布図は、2つの数値の関係（相関）を視覚的に確認するために使います。点の分布パターンを見ることで、関係の強さや傾向（増える・減る）を読み取れます。",
        "折れ線グラフ": "折れ線グラフは、時間などの順序に沿ったデータの変化を追うのに適しています。例えば調査年ごとの推移を見たいときに使います。",
        "棒グラフ": "棒グラフは、カテゴリーごとの数値の違いを比べるときに使います。長さの違いが一目でわかるため、比較に便利です。",
        "円グラフ": "円グラフは、全体に対する割合を表すのに使います。特定の項目が全体の中でどれくらい占めているかを把握するのに適しています。",
        "ヒストグラム": "ヒストグラムは、ある変数の「分布（ばらつき）」を確認するときに使います。多く出現する範囲や偏りを可視化します。",
        "箱ひげ図": "箱ひげ図は、データのばらつきや外れ値を示すために使います。中央値、四分位範囲、最大・最小値が一目でわかります。"
    }

    st.info(graph_explanations.get(graph_type, ""))
        
    if graph_type in ["折れ線グラフ", "棒グラフ", "円グラフ"]:
        col1, col2 = st.columns(2)
        for var, col in zip([x_col, y_col], st.columns(2)):
            with col:
                st.markdown(f"#### {graph_type}：{var}")

                if graph_type == "棒グラフ":
                    fig = px.bar(df, x="調査年", y=var, title=f"{var}（棒グラフ）")
                elif graph_type == "折れ線グラフ":
                    fig = px.line(df, x="調査年", y=var, title=f"{var}（折れ線グラフ）", markers=True)
                elif graph_type == "円グラフ":
                    df_sorted = df.sort_values(var, ascending=False).head(10)
                    fig = px.pie(df_sorted, names="調査年", values=var, title=f"{var}（上位10件・円グラフ）")

                st.plotly_chart(fig, use_container_width=True)

    else:
        if graph_type == "散布図":
            col_left, col_main, col_right = st.columns([1, 2, 1])
            with col_main:
                st.markdown(f"#### 散布図：{x_col} vs {y_col}")
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}（散布図）")
                st.plotly_chart(fig, use_container_width=True)

        elif graph_type == "ヒストグラム":
            for var, col in zip([x_col, y_col], st.columns(2)):
                with col:
                    st.markdown(f"#### ヒストグラム：{var}")
                    fig = px.histogram(df, x=var, nbins=20, title=f"{var} のヒストグラム")
                    st.plotly_chart(fig, use_container_width=True)

        elif graph_type == "箱ひげ図":
            for var, col in zip([x_col, y_col], st.columns(2)):
                with col:
                    st.markdown(f"#### 箱ひげ図：{var}")
                    fig = px.box(df, y=var, title=f"{var} の箱ひげ図")
                    st.plotly_chart(fig, use_container_width=True)



# --- セッション変数の初期化 ---
if "analyze_count" not in st.session_state:
    st.session_state.analyze_count = 0

# --- 解析ボタンの表示と制限 ---
if st.session_state.graph_shown:
    st.markdown("## 回帰分析")

    analyze_max = 5
    remaining = analyze_max - st.session_state.analyze_count
    st.info(f"🧮 残り解析可能回数：**{remaining}回**（全{analyze_max}回まで）")

    # --- 氏名と仮説入力（st.formを使う） ---
    st.subheader("📝 事前入力（必須）")

    with st.form("analysis_form", clear_on_submit=False):
        name = st.text_input("氏名を入力してください", key="name_input_form")
        hypothesis = st.text_area(
            "🔍 この組み合わせの理由や仮説を入力してください", key="hypothesis_input_form", height=100
        )

        analyze_label = f"解析 ×{remaining}"
        submitted = st.form_submit_button(analyze_label)

    # --- 入力チェックと処理 ---
    if remaining <= 0:
        st.warning("⚠️ 解析回数の限界です")
    elif submitted:
        if name.strip() == "" or hypothesis.strip() == "":
            st.warning("※ 氏名と仮説を入力してください。")
        else:
            st.session_state.analyze_count += 1
            st.session_state.analyze_shown = True
            st.session_state.user_name = name
            st.session_state.hypothesis = hypothesis
            st.rerun()


# --- 回帰分析 & ランキング登録 ---
# --- 回帰分析 & ランキング登録 ---
if st.session_state.graph_shown and st.session_state.analyze_shown:
    df_valid = df[["調査年", x_col, y_col]].dropna()
    X = df_valid[[x_col]]
    y = df_valid[y_col]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.session_state.r2 = r2
    st.session_state.x_col = x_col
    st.session_state.y_col = y_col

    fig2 = px.scatter(df_valid, x=x_col, y=y_col, hover_name="調査年", trendline="ols", color=x_col)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""<div style="padding:1rem;border-left:6px solid #00acc1;background:#e0f7fa;">
        <h4>決定係数 R²</h4>
        <p style="font-size:32px;text-align:center;color:#01579b;">{r2:.3f}</p>
    </div>""", unsafe_allow_html=True)

    # --- ランキング登録 ---
    new_record = pd.DataFrame([{
        "氏名": st.session_state.user_name,
        "X": x_col,
        "Y": y_col,
        "R2": r2,
        "仮説": st.session_state.hypothesis
    }])
    RANKING_FILE = "team_ranking.csv"
    if os.path.exists(RANKING_FILE) and os.path.getsize(RANKING_FILE) > 0:
        existing = pd.read_csv(RANKING_FILE)
        updated = pd.concat([existing, new_record], ignore_index=True)
    else:
        updated = new_record
    updated.to_csv(RANKING_FILE, index=False)

    st.success("✅ ランキングに登録されました！")


# --- チームランキング一覧（常時表示、R²は3回以上で表示） ---
# --- チームランキング一覧（常時表示、R²は3回以上で表示） ---
RANKING_FILE = "team_ranking.csv"
if os.path.exists(RANKING_FILE) and os.path.getsize(RANKING_FILE) > 0:
    with st.expander("📋 チームランキング一覧（クリックで表示）", expanded=False):
        st.subheader("📋 チームランキング一覧（R²順）")

        df_rank = pd.read_csv(RANKING_FILE, encoding='utf-8-sig').sort_values("R2", ascending=False)

        # 表示カラム：Yは非表示、R²は解析1回目までのみ表示
        columns_to_show = ["氏名", "X", "仮説"]
        if st.session_state.analyze_count <= 1:
            columns_to_show.insert(2, "R2")  # R2 を仮説の前に表示
        else:
            st.info("※ ここからは決定係数（R²）は非表示です。")

        # 表示用ラベルの調整
        rename_dict = {
            "氏名": "氏名",
            "X": "X軸の項目",
            "Y": "",  # 表示しない
            "R2": "R²値",
            "仮説": "仮説"
        }

        st.dataframe(
            df_rank[columns_to_show].rename(columns=rename_dict),
            use_container_width=True,
            hide_index=True,
            column_config={
                "仮説": st.column_config.TextColumn("仮説", width="large")
            }
        )
else:
    st.info("まだランキング登録がありません。")
