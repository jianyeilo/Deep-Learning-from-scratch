# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import time

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, 
        flatten=True, 
        one_hot_label=False
        )
    return x_test, t_test

#習済みの重みとバイアスを読み込む関数,sample_weight.pkl の中には、すでに訓練されたパラメータが保存されています
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#習済みパラメータを取り出しています。
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0


start = time.perf_counter()

for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1


end = time.perf_counter()
print("処理時間:", end - start, "秒")
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


batch_size = 100 # バッチの数
accuracy_cnt_batch = 0


start = time.perf_counter()

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt_batch += np.sum(p == t[i:i+batch_size])


end = time.perf_counter()
print("処理時間_batch:", end - start, "秒")

print("Accuracy_batch:" + str(float(accuracy_cnt_batch) / len(x)))