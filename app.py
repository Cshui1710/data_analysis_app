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

# 管理者モードフラグとタイマー制御変数
if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False
if "lock_time" not in st.session_state:
    st.session_state.lock_time = None
if "hide_time" not in st.session_state:
    st.session_state.hide_time = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "time_limit_minutes" not in st.session_state:
    st.session_state.time_limit_minutes = None

# --- 管理者モード：URLパラメータ対応 ---
admin_param = st.query_params.get("admin", [None])[0]


if admin_param == "shui1710":
    st.session_state.admin_mode = True
    st.success("✅ URL経由で管理者モードが有効です")

    if "time_limit_minutes" not in st.session_state or st.session_state.lock_time is None:
        time_limit_str = st.text_input("⏱ ランキング入力を許可する時間（半角数字・分単位）", key="admin_time_limit_url")
        if time_limit_str.isdigit():
            minutes = int(time_limit_str)
            st.session_state.time_limit_minutes = minutes
            st.session_state.start_time = datetime.now()
            st.session_state.lock_time = st.session_state.start_time + timedelta(minutes=minutes)
            st.session_state.hide_time = st.session_state.start_time + timedelta(minutes=(2 * minutes) / 3)
            st.info(f"⏳ ランキング登録は {minutes} 分間有効です（{st.session_state.lock_time:%H:%M:%S} に締切）")
else:
    st.session_state.admin_mode = False



# --- 時間チェック ---
now = datetime.now()
lock_input = (
    st.session_state.lock_time is not None and now >= st.session_state.lock_time
)
hide_ranking = (
    st.session_state.hide_time is not None and now >= st.session_state.hide_time
)



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
def precompute_valid_pairs(df, threshold=0.9, cache_file="low_r2_pairs.csv"):
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file).values.tolist()
    
    from itertools import combinations
    valid_pairs = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    for col1, col2 in combinations(numeric_cols, 2):
        sub_df = df[[col1, col2]].dropna()
        if len(sub_df) > 2:
            model = LinearRegression()
            model.fit(sub_df[[col1]], sub_df[col2])
            r2 = r2_score(sub_df[col2], model.predict(sub_df[[col1]]))
            if r2 < threshold:
                valid_pairs.append((col1, col2))
    
    pd.DataFrame(valid_pairs, columns=["X", "Y"]).to_csv(cache_file, index=False)
    return valid_pairs

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
numeric_columns = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]
valid_pairs = precompute_valid_pairs(df)
x_candidates = [col for col in numeric_columns if any(col == x for x, y in valid_pairs)]

# --- タイトル ---
st.title("埼玉県オープンデータ分析体験")

# --- サイドバー設定 ---
with st.sidebar:
    st.markdown("## 🎛 データ設定")
    x_col = st.selectbox("X軸にする項目", x_candidates, key="x")
    x_head = x_col[0]

    # x_colに対応するR²<0.9のY軸候補だけを抽出
    y_candidates_all = [y for x, y in valid_pairs if x == x_col and y[0] != x_head]
    y_candidates = [col for col in numeric_columns if col in y_candidates_all]
    y_col = st.selectbox("Y軸にする項目", y_candidates, key="y")

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




    st.markdown("## 回帰分析")
    analyze_button = st.button("解析")

    if analyze_button:
        st.session_state.analyze_shown = True

# --- 回帰分析表示 ---
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

    fig2 = px.scatter(df_valid, x=x_col, y=y_col, hover_name="調査年", trendline="ols",
                      color=x_col, width=800, height=600,
                      title="📊 回帰直線付き散布図（インタラクティブ）")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div style="background: linear-gradient(to right, #e0f7fa, #ffffff); padding: 1.5rem; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 6px solid #00acc1;">
        <h4 style="color:#003366;">決定係数 R²</h4>
        <p style="font-size:36px; font-weight:900; color:#01579b; text-align:center; letter-spacing:1px;">{r2:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

if "r2" in st.session_state and "x_col" in st.session_state and "y_col" in st.session_state:
    st.subheader("🏆 チームランキング機能")

    if lock_input:
        st.warning("⛔ ランキングの登録時間は終了しました。")
    else:
        team_name = st.text_input("チーム名を入力してください", key="team_input")
        hypothesis = st.text_area("🔍 なぜこの項目の組み合わせで決定係数が高かったと思いますか？", key="hypothesis_input", height=100)

        if st.button("ランキングに登録"):
            if team_name and hypothesis:
                new_record = pd.DataFrame([{
                    "チーム名": team_name,
                    "X": st.session_state.x_col,
                    "Y": st.session_state.y_col,
                    "R2": st.session_state.r2,
                    "仮説": hypothesis
                }])

                RANKING_FILE = "team_ranking.csv"
                if os.path.exists(RANKING_FILE) and os.path.getsize(RANKING_FILE) > 0:
                    existing = pd.read_csv(RANKING_FILE)
                    updated = pd.concat([existing, new_record], ignore_index=True)
                else:
                    updated = new_record

                updated.to_csv(RANKING_FILE, index=False)
                st.success("✅ ランキングに登録しました！")
            elif not team_name:
                st.warning("⚠️ チーム名を入力してください。")
            elif not hypothesis:
                st.warning("⚠️ 仮説を入力してください。")

# --- ランキング表示 ---
RANKING_FILE = "team_ranking.csv"
if os.path.exists(RANKING_FILE) and os.path.getsize(RANKING_FILE) > 0:
    if hide_ranking:
        st.info("👁️‍🗨️ 現在、ランキング一覧は非表示時間帯です。")
    else:
        with st.expander("📋 チームランキング一覧（クリックで表示／非表示）", expanded=False):
            st.markdown("""
                <h3 style='font-size: 28px; color: #0d3b66; margin-top: 0;'>📋 チームランキング一覧（R²順）</h3>
            """, unsafe_allow_html=True)

            df_rank = pd.read_csv(RANKING_FILE).sort_values("R2", ascending=False)

            st.dataframe(
                df_rank,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "仮説": st.column_config.TextColumn("仮説", width="large")
                }
            )