import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle #Pythonオブジェクトを保存・読み込みするライブラリ
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):#画像を表示する
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
 #データを入手　flatten:列ベクトルにする　normalize:softmaxを適応. One-Hot Label = 標準基底ベクトル trueなら列ベクトル、flase ならnumber
img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)

