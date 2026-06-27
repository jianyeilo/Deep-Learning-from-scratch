import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#sigmoid fouction h(x) = { 1+ e^(-x) }^(-1)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()
print(sigmoid(x))
print(1)