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
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# --- タイトル ---
st.title("埼玉県オープンデータ分析体験")

# --- サイドバー設定 ---
with st.sidebar:
    st.markdown("## 🎛 データ設定")
    x_col = st.selectbox("X軸にする項目", numeric_columns, key="x")
    x_head = x_col[0]
    y_col = st.selectbox("Y軸にする項目", [col for col in numeric_columns if col != x_col and col[0] != x_head], key="y")

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
        for var, col in zip([x_col, y_col], [col1, col2]):
            with col:
                st.markdown(f"#### {graph_type}：{var}")
                fig, ax = plt.subplots(figsize=(6, 4))
                if graph_type == "棒グラフ":
                    ax.bar(df["調査年"], df[var])
                    ax.set_ylabel(var)
                    ax.tick_params(axis='x', rotation=90)
                elif graph_type == "折れ線グラフ":
                    ax.plot(df["調査年"], df[var], marker="o")
                    ax.set_ylabel(var)
                    ax.tick_params(axis='x', rotation=90)
                elif graph_type == "円グラフ":
                    df_sorted = df.sort_values(var, ascending=False).head(10)
                    ax.pie(df_sorted[var], labels=df_sorted["調査年"], autopct='%1.1f%%')
                    ax.set_aspect('equal')
                fig.tight_layout() 
                st.pyplot(fig)
    else:
        col1, col2 = st.columns(2)
        if graph_type == "散布図":
            col_left, col_main, col_right = st.columns([1, 2, 1])
            with col_main:
                st.markdown(f"#### 散布図：{x_col} vs {y_col}")
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
        elif graph_type == "ヒストグラム":
            for var, col in zip([x_col, y_col], [col1, col2]):
                with col:
                    st.markdown(f"#### ヒストグラム：{var}")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(df[var].dropna(), kde=True, ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)
        elif graph_type == "箱ひげ図":
            for var, col in zip([x_col, y_col], [col1, col2]):
                with col:
                    st.markdown(f"#### 箱ひげ図：{var}")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(y=df[var], ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)



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
        team_name = st.text_input("チーム名を入力してください", key="team_input")

        if st.button("ランキングに登録"):
            if team_name:
                new_record = pd.DataFrame([{
                    "チーム名": team_name,
                    "X": st.session_state.x_col,
                    "Y": st.session_state.y_col,
                    "R2": st.session_state.r2
                }])

                RANKING_FILE = "team_ranking.csv"
                if os.path.exists(RANKING_FILE) and os.path.getsize(RANKING_FILE) > 0:
                    existing = pd.read_csv(RANKING_FILE)
                    updated = pd.concat([existing, new_record], ignore_index=True)
                else:
                    updated = new_record

                updated.to_csv(RANKING_FILE, index=False)
                st.success("ランキングに登録しました！")

        if os.path.exists("team_ranking.csv") and os.path.getsize("team_ranking.csv") > 0:
            st.subheader("📋 チームランキング一覧（R²順）")
            df_rank = pd.read_csv("team_ranking.csv").sort_values("R2", ascending=False)
            st.dataframe(df_rank)
