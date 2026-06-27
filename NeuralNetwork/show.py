import struct
import numpy as np
from PIL import Image


path = "../dataset/train-images-idx3-ubyte"


with open(path, "rb") as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num, rows, cols)

print(magic)        # 2051
print(num)          # 60000
print(rows, cols)   # 28 28
print(images.shape) # (60000, 28, 28)

# 最初の画像を表示
img = images[0]
pil_img = Image.fromarray(img)
pil_img.show()