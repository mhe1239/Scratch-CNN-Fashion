r"""
taskkill /F /IM python.exe

$scriptContent = @'
cd "C:\Scratch-CNN-Fashion\model"
python -u test_network_fashionfor.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\log\training.log" -Encoding utf8 -Append
'@
$scriptPath = "C:\Scratch-CNN-Fashion\log\bg_run.ps1"
Set-Content -Path $scriptPath -Value $scriptContent -Encoding utf8
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""-WindowStyle Hidden
Get-Content "C:\Scratch-CNN-Fashion\log\training.log" -Wait
"""

# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from flexconvnet import FlexConvNet
from data.mnist import load_mnist

# 1. 데이터 로드 (매번 로드할 필요 없으므로 루프 바깥에 배치)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 공통 네트워크 구조 정의
conv_params = [
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}, 
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False}
]

# 평가할 파일 리스트
file_list = [
'snapshot_epoch_18_tacc_0.993_vacc_0.931_log_(7).pkl',
'snapshot_epoch_17_tacc_0.994_vacc_0.930_log_(7).pkl',
'snapshot_epoch_16_tacc_0.994_vacc_0.929_log_(7).pkl',
'snapshot_epoch_15_tacc_0.989_vacc_0.930_log_(7).pkl',
'snapshot_epoch_14_tacc_0.989_vacc_0.928_log_(7).pkl',
'snapshot_epoch_13_tacc_0.988_vacc_0.925_log_(7).pkl',
'snapshot_epoch_12_tacc_0.974_vacc_0.921_log_(7).pkl',
'snapshot_epoch_11_tacc_0.974_vacc_0.920_log_(7).pkl',
'snapshot_epoch_10_tacc_0.972_vacc_0.921_log_(7).pkl'
]
# file=['snapshot_epoch_11_tacc_0.98_vacc_0.93_log_(6).pkl',
# 'snapshot_epoch_12_tacc_0.98_vacc_0.93_log_(6).pkl',
# 'snapshot_epoch_13_tacc_0.99_vacc_0.93_log_(6).pkl',
# 'snapshot_epoch_14_tacc_0.99_vacc_0.93_log_(6).pkl',
# 'snapshot_epoch_15_tacc_0.99_vacc_0.93_log_(6).pkl',
# 'snapshot_epoch_16_tacc_0.991_vacc_0.931_log_(6).pkl',
# 'snapshot_epoch_17_tacc_0.991_vacc_0.931_log_(6).pkl',
# 'snapshot_epoch_18_tacc_0.992_vacc_0.931_log_(6).pkl',
# 'snapshot_epoch_19_tacc_0.992_vacc_0.931_log_(6).pkl',
# 'snapshot_epoch_20_tacc_0.992_vacc_0.931_log_(6).pkl']
print("flexconvnet_fashion_best(7).pkl")
print("=========================================================")
print(f"{'File Name':<50} | {'Float64':<8} | {'Float16':<8}")
print("=========================================================")

# 2. 파일 리스트를 순회하는 메인 for문
for file_name in file_list:
    
    # 중요: float16 변환 오염을 막기 위해 매 루프마다 깨끗한 인스턴스를 생성하거나 
    # 혹은 원본 파라미터를 새로 주입해야 합니다. (여기선 깔끔하게 매번 생성)
    network = FlexConvNet(input_dim=(1, 28, 28), 
                          conv_param_list=conv_params, 
                          hidden_size_list=[256])
    
    # 해당 에폭의 파일 로드
    network.load_params(file_name)
    
    # [Step A] float64 (원본) 테스트 정확도 계산
    x_test_f64 = x_test.astype(np.float64) 
    acc_f64 = network.accuracy(x_test_f64, t_test)
    
    # [Step B] 네트워크 가중치 및 테스트 데이터를 float16으로 강제 형변환
    x_test_f16 = x_test.astype(np.float16)
    for param in network.params.values():
        param[...] = param.astype(np.float16)
        
    # float16 정확도 계산
    acc_f16 = network.accuracy(x_test_f16, t_test)
    
    # 결과 출력 (이름이 기니까 가독성을 위해 정렬 처리)
    print(f"{file_name:<50} | {acc_f64:<8.4f} | {acc_f16:<8.4f}")

print("=========================================================")