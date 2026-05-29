# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from common.layers import *

class DeepConvNet:
    """정확도 99% 이상의 고정밀 가변형 합성곱 신경망
    기본 구성:
        (conv - relu - conv - relu - pool) * N 번 반복
        -> affine - relu - dropout -> affine - dropout -> softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_list=[
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                     {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                     {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}
                 ], # 가변 리스트 구조로 통일
                 hidden_size=50, output_size=10):
        
        self.conv_param_list = conv_param_list
        self.conv_layer_num = len(conv_param_list)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = []
        self.params = {}
        
        # 합성곱(Conv) 대역 차원 자동 추적 및 가중치/계층 동적 생성
        #각 층을 거칠 때마다 데이터의 가로/세로 크기(current_h, current_w)와 채널 수(current_channels)를 수식으로 계산하여 He 초깃값 분모(n_conv, n_affine)를 자동 계산
        current_channels = input_dim[0]
        current_h = input_dim[1]
        current_w = input_dim[2]
        
        w_idx = 1 # 가중치 인덱스 번호 (W1, W2, ...)
        #기존에는 pre_node_nums에 Weight의 size를 유도하기 위한 숫자를 const로 적었음
        #conv_param_list의 길이만큼 반복
        #0은 input, len(list)+1은 Affine층
        for idx in range(1, self.conv_layer_num + 1):
            c_param = self.conv_param_list[idx - 1]
            f_num = c_param['filter_num']
            f_size = c_param['filter_size']
            f_pad = c_param['pad']
            f_stride = c_param['stride']
            
            #현 단계 Conv 출력 차원 계산
            conv_h = (current_h + 2 * f_pad - f_size) // f_stride + 1
            conv_w = (current_w + 2 * f_pad - f_size) // f_stride + 1
            
            #He 초깃값 스케일 자동 계산 (앞 층의 노드 수)
            n_conv = current_channels * f_size * f_size
            weight_init_scale = np.sqrt(2.0 / n_conv)
            
            #파라미터 생성
            self.params['W' + str(w_idx)] = weight_init_scale * np.random.randn(f_num, current_channels, f_size, f_size)
            self.params['b' + str(w_idx)] = np.zeros(f_num)
            
            #계층 추가
            self.layers.append(Convolution(self.params['W' + str(w_idx)], self.params['b' + str(w_idx)], f_stride, f_pad))
            self.layers.append(Relu())
            
            w_idx += 1
            current_channels = f_num
            current_h = conv_h
            current_w = conv_w
            
            #규칙성 부여: Conv가 2개 쌓일 때마다 Pooling 수행 (2n층, (n>0인 정수))
            if idx % 2 == 0:
                self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
                current_h //= 2
                current_w //= 2

        #  완전연결(Affine) 대역 차원 자동 계산 및 가중치/계층 동적 생성
        # 모든 Conv-Pool을 거치고 나온 최종 Flatten 크기 다시 hidden의 input으로 사용
        pool_output_size = int(current_channels * current_h * current_w)
        
        # Affine 1층 (기존 W7 역할)
        n_affine1 = pool_output_size
        weight_init_scale7 = np.sqrt(2.0 / n_affine1)
        self.params['W' + str(w_idx)] = weight_init_scale7 * np.random.randn(pool_output_size, hidden_size)
        self.params['b' + str(w_idx)] = np.zeros(hidden_size)
        self.layers.append(Affine(self.params['W' + str(w_idx)], self.params['b' + str(w_idx)]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        w_idx += 1
        
        # Affine 2층 - 출력층 (기존 W8 역할)
        n_affine2 = hidden_size
        weight_init_scale8 = np.sqrt(2.0 / n_affine2)
        self.params['W' + str(w_idx)] = weight_init_scale8 * np.random.randn(hidden_size, output_size)
        self.params['b' + str(w_idx)] = np.zeros(output_size)
        self.layers.append(Affine(self.params['W' + str(w_idx)], self.params['b' + str(w_idx)]))
        self.layers.append(Dropout(0.5))
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        #가중치를 갖는 계층만 순회하며 기울기 자동 추출
        grads = {}
        w_idx = 1
        for layer in self.layers:
            if isinstance(layer, Convolution) or isinstance(layer, Affine):
                grads['W' + str(w_idx)] = layer.dW
                grads['b' + str(w_idx)] = layer.db
                w_idx += 1

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
        print(f"매개변수를 다음 경로에 저장했습니다: {file_path}")

    def load_params(self, file_name="params.pkl"):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        #저장된 파일 파라미터를 가중치 계층에 순차적으로 동적 결합
        w_idx = 1
        for layer in self.layers:
            if isinstance(layer, Convolution) or isinstance(layer, Affine):
                layer.W = self.params['W' + str(w_idx)]
                layer.b = self.params['b' + str(w_idx)]
                w_idx += 1
def gradient(self, x, t):
        # 1. Forward & Backward (기존과 동일)
        self.loss(x, t, train_flg=True)
        dout = self.last_layer.backward(1)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 2. 기울기 추출 (기존 루프 구조 유지)
        grads = {}
        for i in range(1, self.conv_layer_num + self.hidden_layer_num + 2):
            # --- 가중치가 있는 계층인 Conv, Affine에서 dW, db 추출 ---
            if 'Conv' + str(i) in self.layers:
                layer = self.layers['Conv' + str(i)]
                # L2 로직: lambda가 0이면 기존과 똑같이 .dW 만 가져옴
                grads['W' + str(i)] = layer.dW + self.weight_decay_lambda * self.params['W' + str(i)]
                grads['b' + str(i)] = layer.db
                
            elif 'Affine' + str(i) in self.layers:
                layer = self.layers['Affine' + str(i)]
                grads['W' + str(i)] = layer.dW + self.weight_decay_lambda * self.params['W' + str(i)]
                grads['b' + str(i)] = layer.db
            
            # --- BatchNorm 파라미터(있는 경우만) 추출 (L2 적용x) ---
            if 'BatchNorm' + str(i) in self.layers:
                grads['gamma' + str(i)] = self.layers['BatchNorm' + str(i)].dgamma
                grads['beta' + str(i)] = self.layers['BatchNorm' + str(i)].dbeta
                
        return grads