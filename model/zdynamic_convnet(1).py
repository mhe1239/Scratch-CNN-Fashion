"""ver2
Dropout 추가
hidden_size_list 추가
dynamic_convnet(1)로 이름 변경>즉, (1)이 과거 로직
-dynamic_convnet에 weight_init_std선택 로직 추가
"""
# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient
class DynamicConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size_list=[100, 100], output_size=10, 
                 weight_init_std='he', use_dropout=False, dropout_ratio=0.5):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.use_dropout = use_dropout
        self.params = {}
        
        # 1. 가중치 초기화 (He 초깃값 로직 적용)
        all_size_list = [pool_output_size] + hidden_size_list + [output_size]
        
        # Conv 층 초기화 (W1)
        # ReLU를 사용하므로 He 초깃값 권장
        n = input_dim[0] * filter_size * filter_size
        self.params['W1'] = np.sqrt(2.0 / n) * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        
        # Affine 층 초기화 (W2, W3...)
        for idx in range(1, len(all_size_list)):
            w_idx = idx + 1
            # 2번 예시의 He 초깃값 로직
            scale = np.sqrt(2.0 / all_size_list[idx-1]) 
            self.params['W' + str(w_idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(w_idx)] = np.zeros(all_size_list[idx])

        # 2. 계층 생성
        self.layers = OrderedDict()
        
        # Conv-Relu-Pool
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], 
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 동적 Affine 은닉층 + Dropout 추가
        self.hidden_layer_num = len(hidden_size_list)
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = idx + 1
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(w_idx)], 
                                                      self.params['b' + str(w_idx)])
            self.layers['Relu_act' + str(idx)] = Relu()
            
            # 하이퍼파라미터에 따라 드롭아웃 계층 동적 추가
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)

        # 출력층
        last_w_idx = self.hidden_layer_num + 1
        self.layers['Affine_out'] = Affine(self.params['W' + str(last_w_idx + 1)], 
                                           self.params['b' + str(last_w_idx + 1)])
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x,train_flg=False):
        for layer in self.layers.values():
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False) # 평가는 무조건 train_flg=False
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t, train_flg=True)
        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # 순전파
        self.loss(x, t, train_flg=True)
        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    def load_params(self, file_name="params.pkl"):
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]