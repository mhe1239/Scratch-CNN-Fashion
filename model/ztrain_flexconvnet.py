# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist # Fashion-MNIST를 불러오는 경로 확인 필요
from dynamic_convnet import DynamicConvNet # 저장한 파일명에 맞춰 변경
from flexconvnet import FlexConvNet # 저장한 파일명에 맞춰 변경
from common.trainer import Trainer
# 1. 데이터 로드 (CNN이므로 flatten=False)
# Fashion-MNIST를 사용한다면 해당 데이터 로더를 사용하세요.
# normalize: 0-255 값을 0-1로/flatten:CNN이니 True하지 않음/one_hot_label
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,flatten=False,one_hot_label=False)
# 2. DynamicConvNet을 위한 아키텍처 설정 (DeepConvNet의 구조를 재현)
# 원하는 만큼 층을 쌓거나 줄일 수 있습니다.

conv_params = [
    # 블록 1
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}, # 28->14
    
    # 블록 2
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}, # 14->7
    
    # 블록 3 (선택) - 데이터가 너무 작아지므로 여기서는 풀링 없이 특징만 강화
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False}
];hidden_size_list = [128, 64] # 은닉층을 두 층으로 나누어 더 정교하게 분류
optimizer_type = 'AdamW';weight_decay = 0.01;learning_rate=0.001
if optimizer_type.lower() == 'adamw':
    # AdamW라면: 네트워크의 L2는 끄고(0), 옵티마이저에 값을 넘김
    net_weight_decay = 0
    opt_weight_decay = weight_decay
    print(f"==> AdamW 감지: Optimizer에서 Weight Decay({weight_decay})를 처리합니다.")
else:
    # 그 외(SGD, Adam 등)라면: 네트워크에서 L2를 켜고, 옵티마이저는 0으로 설정
    net_weight_decay = weight_decay
    opt_weight_decay = 0
    print(f"==> {optimizer_type} 감지: Network에서 L2 Regularization({weight_decay})을 처리합니다.")
# 3. 네트워크 초기화
# use_batchnorm과 use_dropout을 True로 설정하여 DeepConvNet의 성능을 흡수
network = FlexConvNet(input_dim=(1, 28, 28), 
                    conv_param_list=conv_params, 
                    hidden_size_list=[50], 
                    output_size=10,
                    weight_init_std='he',
                    use_groupnorm=True, 
                    use_dropout=True,
                    dropout_ratio=0.3,
                    weight_decay_lambda=net_weight_decay)
optimizer_param = {'lr': learning_rate}
if optimizer_type.lower() == 'adamw':
    optimizer_param['weight_decay'] = opt_weight_decay
# 4. 트레이너 초기화 및 학습 시작
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer=optimizer_type, optimizer_param=optimizer_param,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 5. 매개변수 보관
save_name = "flexconvnet_fashion_params.pkl"
network.save_params(save_name)
print(f"파일이 저장된 절대 경로: {os.path.abspath(save_name)}")
print("Saved flexconvnet Network Parameters!")