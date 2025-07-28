import pandas as pd

# CSVファイルの読み込み
csv_path = "data.csv"
df = pd.read_csv(csv_path)

# 決定係数で降順ソートし、上位3件を抽出 → 逆順（3位→1位）に
top3 = df.sort_values(by="R2", ascending=False).head(3).reset_index(drop=True)
top3 = top3[::-1].reset_index(drop=True)

# 各チームのHTMLをJS用配列にまとめる（プレーンな文字列）
ranks = [
    ("🥉", "第3位"),
    ("🥈", "第2位"),
    ("🥇", "第1位"),
]

ranking_items = []
for i, row in top3.iterrows():
    icon, rank_label = ranks[i]
    html_block = f"""
      `<div class="ranking">
        <h2>{icon} {rank_label}：{row['チーム名']}</h2>
        <p><strong>仮説：</strong>{row['仮説'] if pd.notna(row['仮説']) else '（仮説未記入）'}</p>
        <p><strong>X:</strong> {row['X']} × <strong>Y:</strong> {row['Y']}</p>
        <p><strong>決定係数 R² =</strong> {row['R2']:.4f}</p>
        <p class="celebrate">🎉 おめでとうございます！ 🎊</p>
      </div>`
    """
    ranking_items.append(html_block)

# JavaScriptの配列文字列化
ranking_js_array = "[\n" + ",\n".join(ranking_items) + "\n]"

# HTMLテンプレート全体
html_template = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <title>決定係数ランキング発表</title>
  <style>
    body {{
      background: linear-gradient(to right, #ffe259, #ffa751);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      text-align: center;
      color: #333;
      padding: 50px;
    }}
    h1 {{
      font-size: 3em;
      color: #fff;
      text-shadow: 2px 2px 4px #000;
    }}
    #container {{
    display: flex;
    flex-direction: column; /* ← 逆順にしない */
    align-items: center;
    gap: 30px;
    margin-top: 40px;
    }}
    .ranking {{
      background: white;
      border-radius: 20px;
      padding: 30px;
      width: 80%;
      max-width: 600px;
      box-shadow: 0 10px 20px rgba(0,0,0,0.2);
      display: none;
    }}
    .ranking h2 {{
      font-size: 2em;
      color: #e67e22;
    }}
    .ranking p {{
      font-size: 1.2em;
      margin: 10px 0;
    }}
    .celebrate {{
      font-size: 1.5em;
      color: #d35400;
      font-weight: bold;
      animation: pop 1s infinite alternate;
    }}
    @keyframes pop {{
      0% {{ transform: scale(1); }}
      100% {{ transform: scale(1.1); }}
    }}
    .instruction {{
      margin-top: 20px;
      font-size: 1.2em;
      color: #fff;
    }}
  </style>
</head>
<body>
  <h1>🎉 決定係数ランキング発表 🎉</h1>
  <div class="instruction">クリックまたはEnterキーで上に発表が追加されます</div>

  <div id="container"></div>

  <script>
    const rankings = {ranking_js_array};
    let currentIndex = 0;

    function showNext() {{
      if (currentIndex < rankings.length) {{
        const container = document.getElementById("container");
        const wrapper = document.createElement("div");
        wrapper.innerHTML = rankings[currentIndex];
        const newElement = wrapper.firstElementChild;
        newElement.style.display = "block";
        container.prepend(newElement);
        currentIndex++;
      }}
    }}

    document.addEventListener('click', showNext);
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Enter') showNext();
    }});
  </script>
</body>
</html>
"""

# HTMLをファイルとして保存
output_path = "team_ranking_announcement.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"✅ HTMLファイルを作成しました: {output_path}")
