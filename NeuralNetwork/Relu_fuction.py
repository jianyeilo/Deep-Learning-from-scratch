import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sigmoid_fuction  as sigmoid

def Relu(x):
    return np.maximum(0,x)

x = np.arange(-5.0,5.0,0.1)
t = sigmoid.sigmoid(x)
y = Relu(t)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
print(1)
