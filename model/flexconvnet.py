"""
현재 발전 시킨 DynamicConvNet은 결국 SimpleConvNet기반임
DynamicConvNet에 DeepConvNet의 장점을 흡수함>FlexConvNet
아래는 기존 DynamicConvNet과 비교해 바뀐 점이다
1. 조건부 pooing
Conv-BN-ReLU-Pool의 고정된 방식이었는데 더 깊은 층을 사용하기 위해 layer%2==0일때 pool을 넣는 형식을 취하는 것이 좋기에 T/F flag를 취한다
2. OrderedDict Vs list, OrderedDict
list를 쓰는 DeepConvNet은 append로 쌓기에 순서 보장이 쉽지만 특정 레이어로 접근하려면 index를 정확히 기억해야한다 순서가 보장된 딕셔너리를 통한 고유한 key로 즉시 접근 가능
결국에는 gradient의 backward를 위해 순서 보장이 중요하지만 OrderedDict가 충족한다
3. 드롭아웃의 전략적 배치
DeepConvNet은 마지막 출력층 직전뿐만 아니라 은닉층 사이사이에 드롭아웃을 배치하여 과적합을 더 강력하게 막는다
4. AdamW 추가
5. 추가, L2로직
- 4.AdamW에 따라 AdamW과 L2로직이 작동시 이중으로 적용된다
이에 따라 AdamW일때는 이 로직을 회피하도록 설계한다
6. predict의 x값 수정
x = layer.forward(x)을 x = layer.forward(x,train_flg)로 수정함으로써 추론 모드임을 명시한다
"""
# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *#Norm
#from common.gradient import numerical_gradient
class FlexConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param_list=[
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
                     {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
                     {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
                     {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}
                 ],
                 hidden_size_list=[100], output_size=10, 
                 weight_init_std='he', use_groupnorm=False, use_dropout=True, fc_dropout_ratio=0.3,conv_dropout_ratio=0.1,
                 weight_decay_lambda=0
                 ):#
        
        self.params = {}
        self.layers = OrderedDict()
        self.conv_layer_num = len(conv_param_list)
        #self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_groupnorm=use_groupnorm
        # 1. 차원 추적 (DeepConvNet 방식)
        current_channels, current_h, current_w = input_dim
        
        # 2. 합성곱 계층 생성 루프
        for i, prec in enumerate(conv_param_list):
            idx = i + 1
            # 가중치 초기화 (DeepConvNet의 He 초기화 로직 흡수)
            n_conv = current_channels * prec['filter_size'] * prec['filter_size']
            scale = self.__get_init_scale(weight_init_std, n_conv)
            
            self.params['W' + str(idx)] = scale * np.random.randn(prec['filter_num'], current_channels, prec['filter_size'], prec['filter_size'])
            self.params['b' + str(idx)] = np.zeros(prec['filter_num'])
            
            # 계층 추가
            self.layers['Conv' + str(idx)] = Convolution(self.params['W' + str(idx)], self.params['b' + str(idx)], prec['stride'], prec['pad'], use_batchnorm=self.use_groupnorm)
            
            # BatchNorm 선택적 흡수
            if self.use_groupnorm:
                num_groups = 4
                if prec['filter_num'] % num_groups != 0:# 채널 수가 num_groups로 나누어 떨어지는지 확인
                    num_groups = prec['filter_num'] # 안 나눠지면 전체를 하나로 (LayerNorm 효과)
                self.params['gamma' + str(idx)] = np.ones(prec['filter_num'])
                self.params['beta' + str(idx)] = np.zeros(prec['filter_num'])
                self.layers['GroupNorm' + str(idx)] = GroupNormalization(
                    self.params['gamma' + str(idx)], 
                    self.params['beta' + str(idx)], 
                    group=4
                )
            
            self.layers['Relu_conv' + str(idx)] = Relu()
            
            # 출력 크기 업데이트
            current_h = (current_h + 2*prec['pad'] - prec['filter_size']) // prec['stride'] + 1
            current_w = (current_w + 2*prec['pad'] - prec['filter_size']) // prec['stride'] + 1
            current_channels = prec['filter_num']
            
            # DeepConvNet의 조건부 풀링 전략: 설정에 따라 풀링 수행
            if prec.get('pool', False):
                self.layers['Pool' + str(idx)] = Pooling(pool_h=2, pool_w=2, stride=2)
                current_h //= 2
                current_w //= 2
            if use_dropout and conv_dropout_ratio > 0:
                self.layers['Dropout_conv' + str(idx)] = Dropout(conv_dropout_ratio)
        # 3. 완전연결계층(Affine) 생성 (DeepConvNet의 가변 은닉층 로직)
        pool_output_size = current_channels * current_h * current_w
        all_affine_sizes = [pool_output_size] + hidden_size_list + [output_size]
        self.total_weight_layers = self.conv_layer_num + len(hidden_size_list) + 1
        for i in range(len(all_affine_sizes) - 1):
            idx = self.conv_layer_num + i + 1
            n_affine = all_affine_sizes[i]
            scale = self.__get_init_scale(weight_init_std, n_affine)
            
            self.params['W' + str(idx)] = scale * np.random.randn(all_affine_sizes[i], all_affine_sizes[i+1])
            self.params['b' + str(idx)] = np.zeros(all_affine_sizes[i+1])
            
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            
            # 마지막 출력층 직전까지만 ReLU와 Dropout 추가
            if idx < self.total_weight_layers:
                self.layers['Relu_affine' + str(idx)] = Relu()
                if use_dropout:
                    self.layers['Dropout' + str(idx)] = Dropout(fc_dropout_ratio)
        
        self.last_layer = SoftmaxWithLoss()
    def __get_init_scale(self, weight_init_std, n):
        """layer가 Deep해짐에 따른 Conv와 Affine에서의 __init_weight"""
        if str(weight_init_std).lower() in ('relu', 'he'):
            return np.sqrt(2.0 / n)
        elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
            return np.sqrt(1.0 / n)
        return float(weight_init_std)

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():
            # # Dropout이나 BatchNorm처럼 train_flg가 필요한 레이어와 아닌 레이어 구분
            # if isinstance(layer, (Dropout, BatchNormalization)):
            #     x = layer.forward(x, train_flg)
            # else:
            #     x = layer.forward(x)
            x = layer.forward(x,train_flg)#common.layer의 forward에 나머지 ,train_flg=False룰 모두 추가해 해결
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        # 모든 가중치의 제곱합 계산 (L2 penalty)
        weight_decay = 0
        for idx in range(1, self.total_weight_layers + 1):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t)+ weight_decay
        

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc = 0.0
        # cell: 전체 검증 데이터 개수가 batch_size로 딱 떨어지지 않고 나머지가 남는 경우 방지
        # if 데이터 1005개, 배치 100개인 경우 마지막 5개의 데이터는 정확도 평가에서 완전히 누락
        for i in range(int(np.ceil(x.shape[0] / batch_size))):
            #np.ceil를 씀에 따라 전체 데이터 수(x.shape[0])를 초과하는 인덱싱 슬라이싱시에 비어있지만 python에선 이를 자동으로 처리한다 but, 배치 연산 중에 데이터가 아예 비어있는 상황을 원천 차단하고 가독성을 높이기 위해
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, x.shape[0])
            tx = x[start_idx:end_idx]
            tt = t[start_idx:end_idx]
            y = self.predict(tx, train_flg=False) # 평가는 무조건 train_flg=False
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        loss = self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 2. 기울기 추출 (기존 루프 구조 유지)
        grads = {}
        for i in range(1, self.total_weight_layers + 1):
            # --- 가중치가 있는 계층인 Conv, Affine에서 dW, db 추출 ---
            if 'Conv' + str(i) in self.layers:
                layer = self.layers['Conv' + str(i)]
                # L2 로직: lambda가 0이면 기존과 똑같이 .dW 만 가져옴, dW = dLoss/dW + lambda * W
                grads['W' + str(i)] = layer.dW + self.weight_decay_lambda * self.params['W' + str(i)]
                grads['b' + str(i)] = layer.db
                
            elif 'Affine' + str(i) in self.layers:
                layer = self.layers['Affine' + str(i)]
                grads['W' + str(i)] = layer.dW + self.weight_decay_lambda * self.params['W' + str(i)]
                grads['b' + str(i)] = layer.db
            
            # --- BatchNorm 파라미터(있는 경우만) 추출 (L2 적용x) ---
            gn_key = 'GroupNorm' + str(i)
            if self.use_groupnorm and gn_key in self.layers:
                grads['gamma' + str(i)] = self.layers[gn_key].dgamma
                grads['beta' + str(i)] = self.layers[gn_key].dbeta
                
        return grads, loss
    
    
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

        # 파라미터 로드 후 각 레이어 객체의 W, b 등을 동기화
        for i in range(1, self.total_weight_layers + 1):
            # Conv/Affine 동기화
            if 'Conv' + str(i) in self.layers:
                self.layers['Conv' + str(i)].W = self.params['W' + str(i)]
                self.layers['Conv' + str(i)].b = self.params['b' + str(i)]
            if 'Affine' + str(i) in self.layers:
                self.layers['Affine' + str(i)].W = self.params['W' + str(i)]
                self.layers['Affine' + str(i)].b = self.params['b' + str(i)]
            
            # BatchNorm 동기화
            if 'GroupNorm' + str(i) in self.layers:
                self.layers['GroupNorm' + str(i)].gamma = self.params['gamma' + str(i)]
                self.layers['GroupNorm' + str(i)].beta = self.params['beta' + str(i)]
    def load_params_from_dict(self, params_dict):
        """딕셔너리로부터 가중치를 받아와 실제 레이어 객체들에 주입"""
        for i in range(1, self.total_weight_layers + 1):
            # 1. Conv 처리
            if 'Conv' + str(i) in self.layers:
                self.layers['Conv' + str(i)].W = params_dict['W' + str(i)]
                self.layers['Conv' + str(i)].b = params_dict['b' + str(i)]
                
                # BN 처리: 레이어에도 있고, '불러온 데이터'에도 있을 때만!
                bn_key = 'GroupNorm' + str(i)
                gamma_key = 'gamma' + str(i)
                if self.use_groupnorm and bn_key in self.layers and gamma_key in params_dict:
                    self.layers[bn_key].gamma = params_dict[gamma_key]
                    self.layers[bn_key].beta = params_dict['beta' + str(i)]
            
            # 2. Affine 처리 (이전과 동일)
            elif 'Affine' + str(i) in self.layers:
                self.layers['Affine' + str(i)].W = params_dict['W' + str(i)]
                self.layers['Affine' + str(i)].b = params_dict['b' + str(i)]