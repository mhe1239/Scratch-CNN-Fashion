"""ver3
dynamic_convnet에 weight_init_std선택 로직 추가
weight_init_std='he' or 'xavier' or num
dynamic_convnet에 BatchNormalization 추가
--(2)
conv의 계층 수 자동화
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
                 conv_param_list=[{'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}],# Convolution를 Dynamic하게 하기 위해 list{딕셔너리}로 변경
                 hidden_size_list=[100, 100], output_size=10, 
                 weight_init_std='he', use_dropout=False, dropout_ratio=0.2):
        
        self.input_dim = input_dim
        self.conv_param_list = conv_param_list
        self.conv_layer_num = len(conv_param_list)## Conv 블록 개수
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.output_size = output_size
        self.use_dropout = use_dropout
        self.params = {}
        
        # 1. 가중치 초기화 분리 호출
        self.__init_weight(weight_init_std)

        # 2. 계층 생성
        self.layers = OrderedDict()
        
        # Conv 블록 동적 생성 루프: Conv-Relu-[Norm]-Pool
        for idx in range(1, self.conv_layer_num + 1):
            c_param = self.conv_param_list[idx - 1]
            self.layers['Conv' + str(idx)] = Convolution(self.params['W' + str(idx)], self.params['b' + str(idx)], 
                                                         c_param['stride'], c_param['pad'])
            self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            self.layers['Relu_conv' + str(idx)] = Relu()
            self.layers['Pool' + str(idx)] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 동적 Affine 은닉층 + Dropout 추가
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = self.conv_layer_num + idx  # Conv가 2개였다면 W3, W4...로 이어짐
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(w_idx)], 
                                                      self.params['b' + str(w_idx)])
            self.layers['Relu_affine' + str(idx)] = Relu()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ratio)
        # 출력층 (Affine_out은 가중치 리스트의 마지막 인덱스를 사용)
        last_w_idx = self.conv_layer_num + self.hidden_layer_num + 1  # 맨 마지막 가중치 번호
        self.layers['Affine_out'] = Affine(self.params['W' + str(last_w_idx)], 
                                           self.params['b' + str(last_w_idx)])
        
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """가중치 초기화 로직 구현 (Conv와 Affine 모두 완전 동적화)
        Parameters
        ----------
        weight_init_std : 'he'/'relu', 'xavier'/'sigmoid', 혹은 고정 숫자 수치 값
        """
        # Conv 블록 가중치 초기화 (W1, W2, ...)
        current_channels = self.input_dim[0]  # 최초 입력 채널 (예: 흑백=1)
        current_h = self.input_dim[1]         # 최초 입력 높이 (예: 28)
        current_w = self.input_dim[2]         # 최초 입력 너비 (예: 28)
        #Conv 각 계층에 대해 for로 초기화
        for idx in range(1, self.conv_layer_num + 1):
            c_param = self.conv_param_list[idx - 1]
            f_num = c_param['filter_num']
            f_size = c_param['filter_size']
            f_pad = c_param['pad']
            f_stride = c_param['stride']
            
            # 현 단계 Conv 출력 크기 계산
            conv_h = (current_h + 2*f_pad - f_size) // f_stride + 1
            conv_w = (current_w + 2*f_pad - f_size) // f_stride + 1
            
            # 현 단계 Pooling 출력 크기 계산 (2x2 풀링, 스트라이드 2 고정 가정)
            pool_h = conv_h // 2
            pool_w = conv_w // 2
            
            # He / Xavier 용 노드 수 계산
            n_conv = current_channels * f_size * f_size
            scale_conv = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale_conv = np.sqrt(2.0 / n_conv)
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale_conv = np.sqrt(1.0 / n_conv)
            else:
                scale_conv = float(weight_init_std)
            # 파라미터 할당
            self.params['W' + str(idx)] = scale_conv * np.random.randn(f_num, current_channels, f_size, f_size)
            self.params['b' + str(idx)] = np.zeros(f_num)
            self.params['gamma' + str(idx)] = np.ones(f_num)
            self.params['beta' + str(idx)] = np.zeros(f_num)
            
            # 다음 층을 위해 인풋 채널 및 해상도 정보를 업데이트 업데이트
            current_channels = f_num
            current_h = pool_h
            current_w = pool_w
            
        # 모든 Conv-Pool 블록을 통과한 최종 Flatten 차원 계산
        pool_output_size = int(current_channels * current_h * current_w)
        
        # 완전연결계층 노드 리스트 구성: [전개된 풀링 출력 크기, 은닉층1, 은닉층2, ..., 출력층 크기]
        affine_size_list = [pool_output_size] + self.hidden_size_list + [self.output_size]
        
        for idx in range(1, len(affine_size_list)):
            w_idx = self.conv_layer_num + idx  # Conv 개수 다음 번호부터 부여
            n_affine = affine_size_list[idx - 1]
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale_affine = np.sqrt(2.0 / n_affine)
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale_affine = np.sqrt(1.0 / n_affine)
            else: 
                scale_affine = float(weight_init_std)
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

    def numerical_gradient(self, x, t):#차피 학습시에 안쓰기에 다음부턴 적용x
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
        loss_w = lambda w: self.loss(x, t, train_flg=False)
        
        grads = {}
        for key in self.params.keys():
            # gamma, beta는 수치 미분 연산에서 제외하되, Optimizer와의 Key 일치를 위해 0으로 채움
            if 'gamma' in key or 'beta' in key: 
                grads[key] = np.zeros_like(self.params[key])
                continue
            grads[key] = numerical_gradient(loss_w, self.params[key])
        return grads

    def gradient(self, x, t):
        """오차역전파법 기반 동적 기울기 추출 (Conv 블록 개수 가변 대응)"""
        self.loss(x, t, train_flg=True)#forward
        dout = 1#backward
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        #결과 저장, Conv & Affine
        grads = {}
        # Conv & BatchNorm 블록 기울기 동적 추출
        for idx in range(1, self.conv_layer_num + 1):
            grads['W' + str(idx)] = self.layers['Conv' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Conv' + str(idx)].db
            grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
            grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        
        # Affine 은닉층 기울기 동적 추출
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = self.conv_layer_num + idx
            grads['W' + str(w_idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(w_idx)] = self.layers['Affine' + str(idx)].db
            
        # 출력층 기울기 동적 추출
        last_w_idx = self.conv_layer_num + self.hidden_layer_num + 1
        grads['W' + str(last_w_idx)] = self.layers['Affine_out'].dW
        grads['b' + str(last_w_idx)] = self.layers['Affine_out'].db
        return grads

    def save_params(self, file_name="params.pkl"):
            with open(file_name, 'wb') as f:
                pickle.dump(self.params, f)
    def load_params(self, file_name="params.pkl"):
        if os.path.exists(file_name):#실행 파일과 같은 위치에 있는 경우
            file_path = file_name
        else:#common에 있는 경우
            try:
                file_path = os.path.join(os.path.dirname(__file__), file_name)
            except NameError:
                # __file__이 정의되지 않는 대화형 환경 대피책
                file_path = file_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {file_path}")
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
            
        # 레이어와 파라미터 동적 바인딩 복구
        for idx in range(1, self.conv_layer_num + 1):
            self.layers['Conv' + str(idx)].W = self.params['W' + str(idx)]
            self.layers['Conv' + str(idx)].b = self.params['b' + str(idx)]
            self.layers['BatchNorm' + str(idx)].gamma = self.params['gamma' + str(idx)]
            self.layers['BatchNorm' + str(idx)].beta = self.params['beta' + str(idx)]
        
        for idx in range(1, self.hidden_layer_num + 1):
            w_idx = self.conv_layer_num + idx
            self.layers['Affine' + str(idx)].W = self.params['W' + str(w_idx)]
            self.layers['Affine' + str(idx)].b = self.params['b' + str(w_idx)]
            
        last_w_idx = self.conv_layer_num + self.hidden_layer_num + 1
        self.layers['Affine_out'].W = self.params['W' + str(last_w_idx)]
        self.layers['Affine_out'].b = self.params['b' + str(last_w_idx)]