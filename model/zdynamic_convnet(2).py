"""ver1
dynamic_convnet에 weight_init_std선택 로직 추가
weight_init_std='he' or 'xavier' or num
dynamic_convnet에 BatchNormalization 추가
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
                 weight_init_std='he', use_dropout=False, dropout_ratio=0.2):
        
        self.input_dim = input_dim
        self.conv_param = conv_param
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.use_dropout = use_dropout
        self.params = {}
        
        # 1. 가중치 초기화 분리 호출
        self.__init_weight(weight_init_std)

        # 2. 계층 생성
        self.layers = OrderedDict()
        
        # Conv-Relu-[Norm]-Pool
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], 
                                           conv_param['stride'], conv_param['pad'])
        #  배치 정규화 계층 (차원이 변하지 않으므로 그대로 삽입 가능)
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 동적 Affine 은닉층 + Dropout 추가
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = idx + 1  # W2, W3...
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(w_idx)], 
                                                      self.params['b' + str(w_idx)])
            self.layers['Relu_act' + str(idx)] = Relu()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)

        # 출력층 (Affine_out은 가중치 리스트의 마지막 인덱스를 사용)
        last_w_idx = self.hidden_layer_num + 2  # W1(Conv), W2~W_mid(Hidden), W_last(Output)
        self.layers['Affine_out'] = Affine(self.params['W' + str(last_w_idx)], 
                                           self.params['b' + str(last_w_idx)])
        
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """가중치 초기화 로직 구현
        
        Parameters
        ----------
        weight_init_std : 'he'/'relu', 'xavier'/'sigmoid', 혹은 고정 숫자 수치 값
        """
        filter_num = self.conv_param['filter_num']
        filter_size = self.conv_param['filter_size']
        filter_pad = self.conv_param['pad']
        filter_stride = self.conv_param['stride']
        
        input_size = self.input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        # ----------------------------------------------------
        # 1. Conv 층 (W1) 초기화
        # ----------------------------------------------------
        n_conv = self.input_dim[0] * filter_size * filter_size
        scale_conv = weight_init_std
        
        if str(weight_init_std).lower() in ('relu', 'he'):
            scale_conv = np.sqrt(2.0 / n_conv)
        elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
            scale_conv = np.sqrt(1.0 / n_conv)
            
        self.params['W1'] = scale_conv * np.random.randn(filter_num, self.input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # BatchNorm1을 위한 감마(1로 초기화)와 베타(0으로 초기화)
        self.params['gamma1'] = np.ones(filter_num)
        self.params['beta1'] = np.zeros(filter_num)
        
        # ----------------------------------------------------
        # 2. Affine 층 (W2, W3, ...) 초기화
        # ----------------------------------------------------
        # 완전연결계층 노드 리스트 구성: [전개된 풀링 출력 크기, 은닉층1, 은닉층2, ..., 출력층 크기]
        affine_size_list = [pool_output_size] + self.hidden_size_list + [self.output_size]
        
        for idx in range(1, len(affine_size_list)):
            w_idx = idx + 1  # W2부터 시작
            n_affine = affine_size_list[idx - 1]
            scale_affine = weight_init_std
            
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale_affine = np.sqrt(2.0 / n_affine)
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale_affine = np.sqrt(1.0 / n_affine)
                
            self.params['W' + str(w_idx)] = scale_affine * np.random.randn(affine_size_list[idx - 1], affine_size_list[idx])
            self.params['b' + str(w_idx)] = np.zeros(affine_size_list[idx])

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
            with open(file_name, 'wb') as f:
                pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(file_path, 'rb') as f:
            params = pickle.dump(self.params, f) if not os.path.exists(file_path) else pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
            
        # 복구된 가중치를 실제 생성된 레이어 인스턴스에 동적 동기화
        self.layers['Conv1'].W = self.params['W1']
        self.layers['Conv1'].b = self.params['b1']
        self.layers['BatchNorm1'].gamma = self.params['gamma1']
        self.layers['BatchNorm1'].beta = self.params['beta1']
        
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = idx + 1
            self.layers['Affine' + str(idx)].W = self.params['W' + str(w_idx)]
            self.layers['Affine' + str(idx)].b = self.params['b' + str(w_idx)]
            
        last_w_idx = self.hidden_layer_num + 2
        self.layers['Affine_out'].W = self.params['W' + str(last_w_idx)]
        self.layers['Affine_out'].b = self.params['b' + str(last_w_idx)]