import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

MEMORY_SIZE = 100
CODE_SIZE = 10
STATIC_SIZE = 10

heap_bottom = CODE_SIZE + STATIC_SIZE
stack_top = MEMORY_SIZE

fig, ax = plt.subplots(figsize=(6, 8))

def animate(i):
    global heap_bottom, stack_top

    ax.clear()
    ax.set_ylim(0, MEMORY_SIZE)
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.invert_yaxis()

    event_texts = []

    # ===== ヒープの動き =====
    heap_action = random.random()
    if heap_action < 0.5:
        # ヒープ増加
        heap_bottom += random.randint(1, 3)
        event_texts.append("メモリ確保")
    elif heap_action < 0.75:
        # GC発生
        freed = random.randint(1, 4)
        heap_bottom = max(CODE_SIZE + STATIC_SIZE, heap_bottom - freed)
        event_texts.append(f"GC発生（-{freed}）")
    else:
        # free呼び出し
        freed = random.randint(1, 3)
        heap_bottom = max(CODE_SIZE + STATIC_SIZE, heap_bottom - freed)
        event_texts.append(f"free（-{freed}）")

    # ===== スタックの動き =====
    if random.random() < 0.8:
        # 普通に増加
        stack_top -= random.randint(0, 2)
    else:
        # 🌀 関数終了（再帰や処理完了）
        shrink = random.randint(1, 3)
        stack_top = min(MEMORY_SIZE, stack_top + shrink)
        event_texts.append(f"🔚 関数終了（+{shrink}）")

    # ===== 衝突判定 =====
    if heap_bottom + 1 >= stack_top:
        ax.text(0.5, MEMORY_SIZE / 2, '💥 メモリ不足！', ha='center', va='center', fontsize=16, color='red')
        ani.event_source.stop()

    # ===== 描画 =====
    # コード領域
    ax.fill_between([0, 1], 0, CODE_SIZE, color='gray')
    ax.text(0.5, CODE_SIZE / 2, 'コード領域', ha='center', va='center', fontsize=12, color='white')

    # 静的領域
    ax.fill_between([0, 1], CODE_SIZE, CODE_SIZE + STATIC_SIZE, color='lightgray')
    ax.text(0.5, CODE_SIZE + STATIC_SIZE / 2, '静的領域', ha='center', va='center', fontsize=12)

    # ヒープ領域
    ax.fill_between([0, 1], CODE_SIZE + STATIC_SIZE, heap_bottom, color='skyblue')
    ax.text(0.5, (CODE_SIZE + STATIC_SIZE + heap_bottom) / 2, 'ヒープ領域', ha='center', va='center', fontsize=12)

    # スタック領域
    ax.fill_between([0, 1], stack_top, MEMORY_SIZE, color='pink')
    ax.text(0.5, (stack_top + MEMORY_SIZE) / 2, 'スタック領域', ha='center', va='center', fontsize=12)

    # イベントのテキスト表示
    for j, text in enumerate(event_texts):
        ax.text(0.5, heap_bottom + 3 + j * 5, text, ha='center', va='bottom', fontsize=11, color='green')

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
