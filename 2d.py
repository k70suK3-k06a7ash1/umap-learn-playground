import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_digits

# データ(手書き文字データセット)の読み込み
digits = load_digits()
data = digits.data
target = digits.target

# UMAPの適用
fit = umap.UMAP(random_state=42)
u = fit.fit_transform(data)

# プロットの作成
plt.figure(figsize=(12, 8))

# 各数字ごとに異なる色でプロット
scatter = plt.scatter(u[:, 0], u[:, 1], c=target, cmap='tab10', s=5)

# 凡例の作成
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(i), 
                   markerfacecolor=plt.cm.tab10(i/10), markersize=10)
                   for i in range(10)]
plt.legend(handles=legend_elements, title="Digits", loc="center left", bbox_to_anchor=(1, 0.5))

plt.title('UMAP embedding of the Digits dataset')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

# レイアウトの調整
plt.tight_layout()
plt.show()
