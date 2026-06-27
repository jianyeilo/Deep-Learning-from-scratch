import numpy as np
import matplotlib.pyplot as plt

#Deep Learning の訓練とは、データからモデルを最もよく説明する重み（Weight）とバイアス（Bias）を、最適化アルゴリズムによって学習・推定することである。

# section2  Perceptron
#


def AND(X_1,X_2):
    w_1,w_2,theta = 0.5,0.5,0.7 #w=weight
    tmp = X_1 * w_1 + X_2 * w_2 

    if tmp <= theta :
        return 0
    else :
        return 1
    

print(AND(1,1))

#bias b: theta = -b

x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
print(x*w)