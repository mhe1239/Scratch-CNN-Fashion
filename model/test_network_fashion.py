# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from flexconvnet import FlexConvNet
from data.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
conv_params = [
    {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}
]
network = FlexConvNet(input_dim=(1, 28, 28), 
                      conv_param_list=conv_params, 
                      hidden_size_list=[50])
network.load_params("fashion-deep_convnet_params.pkl")

# sampled = 10000 # 고속화를 위한 표본추출
# x_test = x_test[:sampled]
# t_test = t_test[:sampled]

print("caluculate accuracy (float64) ... ")
print(network.accuracy(x_test, t_test))

# float16(반정밀도)로 형변환
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print("caluculate accuracy (float16) ... ")
print(network.accuracy(x_test, t_test))

"""output
caluculate accuracy (float64) ... 
0.9206
caluculate accuracy (float16) ... 
0.9207
"""