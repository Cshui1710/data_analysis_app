import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import japanize_matplotlib

# Set page config
st.set_page_config(page_title="埼玉データ分析アプリ", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background-color: #f8faff;
    color: #333;
}
section.main > div {
    padding: 2rem;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("saitama_data.csv")
    df = df[df["調査年"] == "2020年度"]
    df = df.drop_duplicates(subset=["地域"])
    df["市町村"] = df["地域"].str.replace("埼玉県 ", "", regex=False)
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df = df.rename(columns={
        "A1101_総人口【人】": "総人口",
        "A1301_15歳未満人口【人】": "15歳未満人口",
        "A1303_65歳以上人口【人】": "65歳以上人口",
        "A4101_出生数【人】": "出生数",
        "A9101_婚姻件数【組】": "婚姻数",
        "E2101_小学校数【校】": "小学校数",
        "E3101_中学校数【校】": "中学校数",
        "E4101_高等学校数【校】": "高校数",
        "F1101_労働力人口【人】": "労働力人口"
    })
    for col in ["総人口", "15歳未満人口", "65歳以上人口", "出生数", "婚姻数",
                "小学校数", "中学校数", "高校数", "労働力人口"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Load
st.title("🧠 埼玉県オープンデータ分析アプリ")
st.markdown("""
白と青を基調とした近未来風のデザインで、グラフ可視化と回帰分析を体験しよう。
""")

# Load data
df = load_data()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# サイドバーで項目選択
with st.sidebar:
    st.markdown("## 🎛 データ設定")
    x_col = st.selectbox("X軸にする項目", numeric_columns, key="x")
    y_col = st.selectbox("Y軸にする項目", [col for col in numeric_columns if col != x_col], key="y")
    graph_type = st.radio("📊 表示するグラフの種類", [
        "散布図", "折れ線グラフ", "棒グラフ", "円グラフ", "ヒストグラム", "箱ひげ図"
    ], key="graph")
    show_graph = st.button("📈 グラフ化")

# メイン画面で表示
if show_graph:
    st.markdown("## 📊 グラフ表示")

    if graph_type in ["折れ線グラフ", "棒グラフ", "円グラフ"]:
    col1, col2 = st.columns(2)

    for var, col in zip([x_col, y_col], [col1, col2]):
        with col:
            st.markdown(f"#### {graph_type}：{var}")
            fig = plt.figure(figsize=(6, 4))

            if graph_type == "棒グラフ":
                plt.bar(df["市町村"], df[var])
                plt.xticks(rotation=90)
                plt.ylabel(var)

            elif graph_type == "折れ線グラフ":
                plt.plot(df["市町村"], df[var], marker="o")
                plt.xticks(rotation=90)
                plt.ylabel(var)

            elif graph_type == "円グラフ":
                df_sorted = df.sort_values(var, ascending=False).head(10)
                plt.pie(df_sorted[var], labels=df_sorted["市町村"], autopct='%1.1f%%')

            st.pyplot(fig)





    else:
        fig = plt.figure(figsize=(8, 6))
        if graph_type == "散布図":
            sns.scatterplot(data=df, x=x_col, y=y_col)
        elif graph_type == "ヒストグラム":
            sns.histplot(df[x_col], kde=True)
        elif graph_type == "箱ひげ図":
            sns.boxplot(data=df[[x_col, y_col]])
        st.pyplot(fig)


    st.markdown("## 🔍 回帰分析")
    X = df[[x_col]].dropna()
    y = df[y_col].dropna()
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # === 装飾付きの回帰式・R²値表示 ===
    st.markdown(f"""
    <div style="background-color:#e6f0ff;padding:1.5rem;border-radius:10px;
                border-left:5px solid #3399ff;margin-bottom:1rem;">
        <h4 style="color:#003366;">📐 回帰式</h4>
        <p style="font-size:18px;">y = <b>{model.coef_[0]:.3f}</b> * x + <b>{model.intercept_:.3f}</b></p>
        <h4 style="color:#003366;">🎯 決定係数 R²</h4>
        <p style="font-size:18px;"><b>{r2:.3f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # === アニメーション的に見えるインタラクティブ回帰散布図 ===
    fig2 = px.scatter(df, x=x_col, y=y_col, hover_name="市町村", trendline="ols",
                      color=x_col,
                      title="📊 回帰直線付き散布図（インタラクティブ）",
                      width=800, height=600)
    st.plotly_chart(fig2, use_container_width=True)

    # ランキング登録
    st.subheader("🏆 チームランキング機能")
    team_name = st.text_input("チーム名を入力してください")
    RANKING_FILE = "team_ranking.csv"

    if st.button("ランキングに登録"):
        if team_name:
            new_record = pd.DataFrame([{
                "チーム名": team_name,
                "X": x_col,
                "Y": y_col,
                "R2": r2
            }])

            if os.path.exists(RANKING_FILE):
                existing = pd.read_csv(RANKING_FILE)
                updated = pd.concat([existing, new_record], ignore_index=True)
            else:
                updated = new_record

            updated.to_csv(RANKING_FILE, index=False)
            st.success("ランキングに登録しました！")

    if os.path.exists(RANKING_FILE):
        st.subheader("📋 チームランキング一覧（R²順）")
        df_rank = pd.read_csv(RANKING_FILE).sort_values("R2", ascending=False)
        st.dataframe(df_rank)
