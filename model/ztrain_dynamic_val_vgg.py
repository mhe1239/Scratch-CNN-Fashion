"""
train_flexconvnet와 차이: 검증 데이터 도입

"""

# coding: utf-8
import sys, os, time
from datetime import timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist # Fashion-MNIST를 불러오는 경로 확인 필요
from flexconvnet import FlexConvNet # 저장한 파일명에 맞춰 변경
from common.trainer import Trainer
from common.util import shuffle_dataset
start_time = time.time()# 시간 측정 시작
# 1. 데이터 로드 (CNN이므로 flatten=False)
# Fashion-MNIST를 사용한다면 해당 데이터 로더를 사용하세요.
# normalize: 0-255 값을 0-1로/flatten:CNN이니 True하지 않음/one_hot_label
(x_train_all, t_train_all), (x_test, t_test) = load_mnist(normalize=True,flatten=False,one_hot_label=False)


x_train_all, t_train_all = shuffle_dataset(x_train_all, t_train_all)
va_rate = 0.2 # 검증 데이터 비율 20%
va_num = int(x_train_all.shape[0] * va_rate)
x_val = x_train_all[:va_num];t_val = t_train_all[:va_num]
x_train = x_train_all[va_num:];t_train = t_train_all[va_num:]

print(f"Data Split - Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {t_test.shape[0]}")

conv_params = [
    # 블록 1: 28x28 -> 14x14 (채널 32)
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    
    # 블록 2: 14x14 -> 7x7 (채널 64)
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    
    # 블록 3: 7x7 -> 3x3 (채널 128)
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False}
]
#optimizer 처리
optimizer_type = 'AdamW'
weight_decay = 1e-6
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

#나머지 Hyper Parameters
hidden_size_list = [256, 128] # 은닉층을 두 층으로 나누어 더 정교하게 분류
#배치 사이즈를 k배 키우면, 학습률도 정확히 k배 키운다>batch_size를 2배 키우면 lr을 1.4~2배 키움
learning_rate=0.0005
batch_size=256
epoch=25
dropout_ratio=0.3
network = FlexConvNet(input_dim=(1, 28, 28), 
                         conv_param_list=conv_params, 
                         hidden_size_list=hidden_size_list, 
                         output_size=10,
                         weight_init_std='he',
                         use_groupnorm=True, 
                         use_dropout=True,
                         dropout_ratio=dropout_ratio,
                         weight_decay_lambda=net_weight_decay)
optimizer_param = {'lr': learning_rate}
if optimizer_type.lower() == 'adamw':
    optimizer_param['weight_decay'] = opt_weight_decay
# 4. 트레이너 초기화 및 학습 시작
trainer = Trainer(network, x_train, t_train, x_val, t_val,
                  epochs=epoch, mini_batch_size=batch_size,
                  optimizer=optimizer_type, optimizer_param=optimizer_param,
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

final_loss = trainer.train_loss_list[-1]
final_val_acc = network.accuracy(x_val, t_val)
final_test_acc = network.accuracy(x_test, t_test)
print("\n" + "="*40)
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Val Acc (Best): {max(trainer.test_acc_list):.4f}")
print(f"Final Test Acc (Real): {final_test_acc:.4f}")
print("="*40)

# 6. 그래프 그리기
train_acc_list = trainer.train_acc_list
val_acc_list = trainer.test_acc_list # Trainer 내부의 test_acc_list는 사실 val_acc입니다.
markers = {'train': 'o', 'val': 's'}
x = np.arange(len(train_acc_list))
plt.figure(figsize=(10, 6)) # 그래프 크기 설정
plt.plot(x, train_acc_list, marker='o', label='train_acc', markevery=1)
plt.plot(x, val_acc_list, marker='s', label='val_acc', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.title(f"FlexConvNet: {optimizer_type} (dropout={dropout_ratio}, wd={weight_decay})")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5) # 그리드 추가로 가독성 향상
plt.show()


# 5. 매개변수 보관
save_name = "flexconvnet_fashion_params.pkl"
network.save_params(save_name)
print(f"파일이 저장된 절대 경로: {os.path.abspath(save_name)}")
print("Saved flexconvnet Network Parameters!")

# =============== 총 소요 시간 출력 ===============
end_time = time.time()
elapsed_seconds = end_time - start_time
td = timedelta(seconds=elapsed_seconds)
time_str = str(td)
if '.' in time_str:
    time_str = time_str[:time_str.find('.') + 4]
else:
    time_str += '.000'

print("============================================================")
print(f"Total Elapsed Time: {time_str}")
print("============================================================")