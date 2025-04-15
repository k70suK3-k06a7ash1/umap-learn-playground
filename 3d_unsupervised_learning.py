import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
from sklearn.datasets import load_digits

# データの読み込み
digits = load_digits()
data = digits.data

# 3次元UMAPの適用
fit = umap.UMAP(n_components=3, random_state=42)
u = fit.fit_transform(data)

# 3Dプロットの作成
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 各数字ごとに異なる色でプロット
scatter = ax.scatter(u[:, 0], u[:, 1], u[:, 2], cmap='tab10', s=10)

# 凡例の作成
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(i), 
                   markerfacecolor=plt.cm.tab10(i/10), markersize=10)
                   for i in range(10)]
ax.legend(handles=legend_elements, title="Digits", loc="center left", bbox_to_anchor=(1.1, 0.5))

ax.set_title('3D UMAP embedding of the Digits dataset')
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')

# レイアウトの調整
plt.tight_layout()
plt.show()
