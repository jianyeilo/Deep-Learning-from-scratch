import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
Neural Network を数学の言葉で整理
"""

# ==========================================
# 1. Neural Network とは
# ==========================================

# ニューラルネットワークは写像（Function / Mapping）

# f : X -> Y

# 例（MNIST）
# f : R^784 -> R^10

# 入力
# x ∈ R^784

# 出力
# y ∈ R^10


# ==========================================
# 2. 1層の数式
# ==========================================

# y = σ(Wx + b)

# W : Weight（重み）
# b : Bias（バイアス）
# σ : Activation Function（活性化関数）

# Wx + b
# = Affine Map（アフィン写像）

# σ
# = Nonlinear Map（非線形写像）

# つまり

# Affine Map
#      ↓
# Nonlinear Map


# ==========================================
# 3. ネットワーク全体
# ==========================================

# f = f_L ○ f_(L-1) ○ ... ○ f_1

# = Composition of Maps（写像の合成）

# 深くなるほど
# より複雑な関数を表現できる。


# ==========================================
# 4. 全単射なのか？
# ==========================================

# 一般には NO

# 単射（Injective）
# f(x)=f(y) ⇒ x=y
# ×

# 全射（Surjective）
# 出力空間全体を覆う
# ×

# 全単射（Bijective）
# ×

# （可逆ニューラルネットワークは例外）


# ==========================================
# 5. 学習（Training）
# ==========================================

# Neural Network

# f(x; W, b)

# 学習とは

# Weight（W）
# Bias（b）

# をデータから最適化して求めること。

# 数学的には

# (W,b)=argmin L(W,b)

# Loss Function（損失関数）
# を最小化する最適化問題。


# ==========================================
# 6. 数学との対応
# ==========================================

# 集合論
#   ↓
# 写像 f : X → Y

# 線形代数
#   ↓
# 行列 W
# ベクトル x
# アフィン写像 Wx+b

# 微分積分
#   ↓
# Gradient（勾配）
# 偏微分

# 最適化
#   ↓
# Gradient Descent
# Adam

# 解析学
#   ↓
# 関数近似
# 連続写像

# 確率論
#   ↓
# データ分布
# 確率モデル

# 数値計算
#   ↓
# GPUによる大規模行列演算


# ==========================================
# 7. 本質
# ==========================================

"""
Neural Network =

アフィン写像（Wx+b）と
非線形写像（Activation Function）
を何層も合成した多変数写像。

Training =

データから
Weight と Bias を最適化し、

最も良い写像 f を構築すること。
"""

# activation fouction - step fouction

def step_function(x):
    if x > 0:
        return 1
    else :
        return 0
    
def step_type(x):
    y = x > 0 #bool 判定
    return y.astype(int) # change type

def step_fouction_array(x):
    return np.array(x>0,dtype=int)

x = np.arange(-5.0 , 5.0, 0.1)
y = step_fouction_array(x)
plt.plot(x,y)
plt.xlim(-5,5)
plt.ylim(-0.1,1.1)
plt.show()

