"""log
Trainer class없이 더 자유롭게 로직을 추가하기 위해 Trainer을 사용하지 않음
1. pooling 적용 여부
DeepConvNet에선 Deep하기에 pooling을 매 계층이 아닌 2계층마다 씀 이를 위해 flag로 로직 추가
2. L2로직
AdamW에 따라 AdamW과 L2로직이 작동시 이중으로 적용된다 이에 따라 AdamW일때는 이 로직을 회피하도록 설계한다
이를 위해 'AdamW'와 같이 설정하게 하고 이에 따라 조건 분기를 적용
3. 검증 데이터 로직 추가
학습 도중에 테스트 데이터(쓰면 치팅)가 아닌 검증 데이터를 보며 최적의 시점을 찾는 것이 정석
또한 train과 train에서 나눈 val의 acc그래프 추가
4. VGG 적용
VGG를 기반으로 conv_params을 수정
5. lr을 직접 바꾸는 로직
train시에 더이상 모델이 학습을 못하는 시점이 오면 lr을 적게 설정해줘야함 단순히 lr을 바꾸면 실제론 적용이 안되기에 optimizer.lr으로 바꾸도록 로직 추가
- 특정 epoch동안 val이 나아지지 않으면 lr을 깎음/특정 epoch동안 더이상 내려가지 않으면 stop함
6. Gradient Clipping 추가
비선형 함수에서 미분 값이 매우 크거나 작아지면 마치 가파른 언덕이고 이는 역전파시 여러개의 큰 가중치를 곱한 후 발생하기에 모든 가중치(W,b)의 기울의 L2 Norm을 계산해 너무 크면 방향은 유지하고 길이를 1로 줄이는 로직 추가
7. 시간 측정
학습 시 총 얼마나 걸렸는지 매 epoch과 총 걸린 시간을 출력하도록 추가
-이후 .show의 경우 이 창이 닫힐때까지가 시간을 기록하기에 .show 로직 전에 하도록 변경
--그냥 .show를 빼고 plt.savefig으로 저장하도록 변경
8. 메모리 폭발(Memory Explosion) 해결
기존 로직은 network.loss(x_val, t_val)를 호출할 때, 검증 데이터 12,000장을 한꺼번에 CNN 레이어에 넣었기에 20GB가 넘어 터졌음
따라서 batch단위로 loss를 계산하도록 get_loss_in_batches 로직을 추가함
9. 빠른 학습을 위한 gradient의 return 값 변경
network.gradient(x_batch, t_batch)를 호출하면 내부적으로 loss를 계산하기 위해 forward가 한 번 실행, 근데 다음 줄에서 network.loss(x_batch, t_batch)를 또 호출하면 Convolution 연산이 한 번 더 수행함 이에 따라 FlexConvNet의 gradient 함수를 살짝 수정해서 기울기와 손실값을 동시에 반환하도록 변경
10. 초반에는 lr을 크게하고 이후에는 작게하는 로직 추가/Warm-up 추가
이를 위해 Cosine Annealing을 추가/초반에 넘 튈 수 있기에 첫 1에폭 동안만 0에서 base_lr까지 선형적으로 올리는 Warm-up로직 추가 get_warmup_cosine_lr를 만듦
11. 매 에폭마다 데이터를 셔플하는 로직 추가
처음에만 셔플해서 변향을 없애지만 우리가 계속 같은 테스트를 보게하면 문제 번호와 정답을 매핑해 외우며 이는 머신러닝에도 적용된다고 생각되어 한 에폭은 전체 데이터를 훑음 그러면 매 에폭마다 섞자!해서 for i in range(iter_per_epoch):마다 섞도록 함
12. 가장 좋은 모델 pkl파일로 저장하기
학습은 끝까지 가되 가장 val_acc가 좋은 파일을 code가 모두 끝났을때 flexconvnet_fashion_best.pkl 이름으로 저장하도록 로직을 짬
-이후 매번 돌릴때마다 이름을 변경하지 않도록 모델 1번 돌리면 log_num.txt파일의 숫자를 파일 이름 뒤에 붙이고 붙여지면 log_num.txt파일의 숫자가 1증가하도록하고 이를 plt.savefig에도 적용함
13. .pkl로 학습 후 error해결
.pkl 파일을 만든 후에 기존 하이퍼파라미터와 conv_params를 모르면 network.load_params("flexconvnet_fashion_best.pkl")시에 error가 나기에 매 학습 실행할때마다 현재 하이퍼파라미터와 conv_params계층 상태가 어떤지 출력하도록 함 summarize_results
14. dorpout의 %를 따로 적용시키기
CNN과 FC(완전연결)은 구분되었다고 할 수 있는데 CNN때엔 이미지를 학습해야하니까 낮게 주고 FC는 우리 인간처럼 높게 줘야할 수 있다고 생각하여 fc_dropout_ratio와 conv_dropout_ratio로 구분하도록 추가함
이 당시 16에폭시에 TrainAcc: 0.8830 | ValAcc: 0.8812을 달성(당시 최고 Acc)
15. checkpoint 추가
돌리던 중 컴퓨터를 끄거나 taskkill /F /IM python.exe 후 다시 모델 돌리면 처음부터 다시 해야하기에 매 에폭마다 체크포인트로써 pkl파일을 추가하고 이후 돌아와서 flex_checkpoint.pkl 파일이 있으면 그때의 에폭과 학습 당시 값들을 확인해 그 포인트부터 시작하도록 함 즉, 처음부터 하는 게 아니라 기존에 저장된 가중치를 불러옴
16. 백그라운드로 코드 돌리기/.log
vscode를 끄면 그대로 돌던 코드가 끝나기에 파워셀에서 백그라운드로 돌리는 코드를 알아내어 적용 동시에 본래는 출력을 common/log.txt에 했는데 이를 training.log에 출력하도록 함(수정이 안되기에 좋음,common/log.txt w모드로 했어서 매 실행마다 사라졌는데 a모드로 했으면 어땠을까 생각한다)
실수1) minst로 학습하고 있었음..
지금까지 from dataset.mnist import load_mnist 즉, mnist로 했었음 from data.mnist import load_mnist 즉, Fashion-MNIST로 변경함
17. train과 test를 함께!
train로직으로 모두 끝내어 pkl파일을 생성하면 이것으로 test를 바로 하고 끝나는 로직을 추가
이후 같은 구조의 pt를 추가하기 시작함
실수2)
이후 DeepConvNet와 같은 구조로 돌렸는데도 이상해서 알아보다가 scratch으로써 NumPy로 구현된 Convolution과 BatchNorm은 연산 과정에서 아주 미세한 반올림 오차가 생긴다는 점을 알아내고 Conv는 쩔 수 없지만 BatchNorm이 문제인가하여 BatchNrom은 False, AdamW을 Adam으로 변경 및 L2를 0으로 해 돌리니 Epoch 07일때 TrainAcc: 0.9460 | ValAcc: 0.9158임을 확인함
이후부턴 BatchNorm의 flag는 False로 고정하도록 함
18. lr을 변경하는 아이디어
현재는 Epoch%3==0마다 val_acc가 줄지 않으면 lr을 깎음 근데 좀 부족한 것 같음. 한 에폭은 배치마다 loss를 저장하긴하는데 이것마다 하면 lr을 빠르게 0애 가까워 질 것임 그러면 현재 1 Epoch마다 3번 loss output하긴 하는데 이도 lr튜닝에 사용하면 좋지 않을까 생각함
이 Current Batch Loss를 토대로 학습 도중 갑자기 loss가 2배로 튀어오르면, Current Batch Loss가 3번될때동안 유의미하게 줄지 않으면 lr을 깎도록 로직을 추가함 이가 바로 lr전략1,2)이다
중간 점검) 에폭 32에서 Train Acc: 1.0000 즉, 오버피팅임을 확인함 이게 오버피팅 직전 가장 잘 학습이 잘 되는지점을 찾아내면 Win이며 이는 val_acc가 가장 높일 때이다 이후 하이퍼파라미터 바꾸면서 돌리다보니 val_acc가 93%를 넘음
한 동안 VGG를 안쓰다가 이때 VGG로직을 써봤는데 이 친구는 92%가 나왔음
3층 와이드 (Best 4): 93.13%;6층 VGG-Lite (Best 5): 92.57%
19. 전체 학습 데이터(Train Data) 유실 방지 및 하드웨어 연산 효율을 위한 batch_size 최적화
기본적으로 batch_size는 2^n으로 하는 것이 연산면에서 좋기에 256으로 했었다 근데 batch_size를 256에서 240으로 변경한다
현재 전체 MNIST 데이터(60,000장) 중 20%(12,000장)를 검증 데이터(Validation)로 분리함에 따라, 순수 학습 데이터 크기(train_size)는 48,000장이 된다. 
이에 따라 iter_per_epoch = max(train_size // batch_size, 1)=현재 코드에선 나머지 버림 연산(//)으로 인해 48000//256=187인데 버림하지 않으면 187.5이다 즉, 매 에폭마다 0.5에 해당하는 128개가 버려진다 이를 위해 train_size/batch_size가 0으로 떨어지게 하기 위해선 240 or 300(자투리 연산 오버헤드가 발생)을 사용해야하는데 CPU의 병렬 벡터화 연산 장치(SIMD)에선 2^n단위로 처리한다
이에 따라 240이 더 효율적이기에 240을 채용한다(48000/240=200)
이에 따라 전체 데이터를 사용하는 경우엔 240이나 480을 사용한다 60000/240=250, /480=125

"""

r"""
$scriptContent = @'
cd "C:\src(3-1)\ArtificialIntelligence\w99"
python -u NoTrainerTrain.py *>&1 | Out-File -FilePath "C:\src(3-1)\ArtificialIntelligence\w99\training.log" -Encoding utf8 -Append
'@
$scriptPath = "C:\src(3-1)\ArtificialIntelligence\w99\bg_run.ps1"
Set-Content -Path $scriptPath -Value $scriptContent -Encoding utf8
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" -WindowStyle Hidden
Get-Content "C:\src(3-1)\ArtificialIntelligence\w99\training.log" -Wait

"""
# coding: utf-8
import sys, os, time, pickle
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.mnist import load_mnist 
from flexconvnet import FlexConvNet 
from common.optimizer import *
from common.util import shuffle_dataset

## 시스템 설정 및 유틸리티 함수
def _get_and_update_log_num(log_num_path):
        """pkl,png 파일 번호 매기는 함수"""
        # 1. 파일이 존재하면 읽고, 없으면 0으로 시작
        if os.path.exists(log_num_path):
            with open(log_num_path, 'r', encoding='utf-8') as f:
                try:
                    current_num = int(f.read().strip())
                except ValueError:
                    current_num = 0  # 파일 내용이 비어있거나 숫자가 아니면 0으로 초기화
        else:
            current_num = 0

        # 2. 다음 실행을 위해 번호를 1 증가시켜 파일에 저장
        next_num = current_num + 1
        with open(log_num_path, 'w', encoding='utf-8') as f:
            f.write(str(next_num))
            
        # 3. 현재 실행에서 사용할 번호(n) 반환
        return current_num

def get_warmup_cosine_lr(current_iter, total_iters, base_lr, warmup_iters=500, min_lr=1e-6):
    """코사인 그래프에 따라 학습률을 계산하는 함수, warmup"""
    # 1. Warm-up 구간 (초반에는 아주 작게 시작해서 선형 증가)
    if current_iter < warmup_iters:
        return min_lr + (base_lr - min_lr) * (current_iter / warmup_iters)
    
    # 2. Cosine Annealing 구간 (이후 서서히 감소)
    current_iter -= warmup_iters
    total_iters -= warmup_iters
    cos_inner = (np.pi * current_iter) / total_iters
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(cos_inner))

def summarize_results(params, final_loss=None, final_acc=None):
    """실험 세팅, CNN 구조 및 결과를 정렬된 텍스트로 요약 출력."""
    use_bn = params.get('use_batchnorm', True)
    fc_dropout_ratio = params.get('fc_dropout_ratio', 0.0)  
    conv_dropout_ratio = params.get('conv_dropout_ratio', 0.0)  
    activation = params.get('activation', 'relu')
    weight_init = params.get('weight_init_std', 'relu')
    bn_status = 'T' if use_bn else 'F'
    # 1. 기본 하이퍼파라미터 필드 구성
    fields = [
        f"Optimizer: {params.get('optimizer', 'N/A'):<7}",
        f"LR: {params.get('lr', 'N/A'):<7}",
        f"Batch: {params.get('batch_size', 0):>4}",
        f"Epochs: {params.get('max_epochs', 0):>3}",
        f"Iters: {params.get('max_iterations', 0):>4}",
        f"Depth(FC): {len(params.get('hidden_size_list', [])):>1}",
        f"Act/Init: {activation}/{weight_init:<5}",
        f"L2: {params.get('weight_decay_lambda', 0):<7}",
        f"BN: {bn_status}",
        f"ConvDrop: {conv_dropout_ratio:<5.1f}",
        f"FcDrop: {fc_dropout_ratio:<5.1f}"
    ]
    setting_str = ", ".join(fields)
    conv_lines = []
    conv_list = params.get('conv_param_list', [])
    if conv_list:
        conv_lines.append("\n  [ CNN Architecture Flow ]")
        conv_lines.append("  " + "-" * 55)
        conv_lines.append("  | Layer | Filter Num | Filter Size | Pad | Stride | Pool  |")
        conv_lines.append("  " + "-" * 55)
        for i, c in enumerate(conv_list):
            pool_str = 'T' if c.get('pool', False) else 'F'
            conv_lines.append(
                f"  | C_{i+1:<3} | {c.get('filter_num', 0):^10} | {c.get('filter_size', 0):^11} "
                f"| {c.get('pad', 0):^3} | {c.get('stride', 1):^6} | {pool_str:^5} |"
            )
        conv_lines.append("  " + "-" * 55)
    conv_str = "\n".join(conv_lines)
    #결과 지표 병합
    if final_loss is not None and final_acc is not None:
        result_str = f" \n>>> [RESULT] | TRAIN LOSS: {final_loss:.6f} | Val ACC: {final_acc:.4f} |"
        return f"{setting_str}\n{conv_str}{result_str}"
    return f"{setting_str}\n{conv_str}"

def get_loss_in_batches(network, x, t, batch_size=128):
    """batch_size만큼 data를 쪼개서 연산이 끝나면 메모리를 비움"""
    #데이터가 클 때 배치를 나누어 Loss의 평균을 구하는 함수로 이거 없이하다가 변수가 20GB 이상을 저장해서 터졌음..
    
    loss_sum = 0
    max_count = x.shape[0]
    for i in range(0, max_count, batch_size):
        x_batch = x[i:i+batch_size]
        t_batch = t[i:i+batch_size]
        loss_sum += network.loss(x_batch, t_batch) * x_batch.shape[0]
    return loss_sum / max_count

## 시간 측정
start_time = time.time()
print("================ Training Started ================")



## Hyperparameter 및 아키텍처 설정
# Hyperparameter
conv_params = [
    # 블록 1: 28x28 해상도 유지하며 특징 추출 (채널 32)
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},  # 28 -> 14
    
    # 블록 2: 특징 심화 (채널 64)
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},  # 14 -> 7
    
    # 블록 3: 고차원 특징 추출 (채널 128)
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False} # 7유지
]
learning_rate = 0.00015
hidden_size_list = [256] 
batch_size = 240
max_epochs = 100
optimizer_type = 'AdamW'
weight_decay = 0.01
min_lr = 1e-6
batchnorm=False
fc_dropout_ratio=0.4;conv_dropout_ratio=0
# Optimizer 가중치 감쇠(Weight Decay) 처리 분기
if optimizer_type.lower() == 'adamw':
    net_weight_decay = 0
    opt_weight_decay = weight_decay
    
    print(f"==> AdamW 감지: Optimizer에서 Weight Decay({weight_decay})를 처리합니다.")
else:
    net_weight_decay = weight_decay
    opt_weight_decay = 0
    print(f"==> {optimizer_type} 감지: Network에서 L2 Regularization({weight_decay})을 처리합니다.")

# 데이터 로드/전처리, 데이터 준비 및 모델 초기화, (CNN이므로 flatten=False)

(x_train_all, t_train_all), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
# 데이터 셔플 및 검증 데이터(20%) 분리
x_train_all, t_train_all = shuffle_dataset(x_train_all, t_train_all)
va_rate = 0.2;va_num = int(x_train_all.shape[0] * va_rate)
x_val = x_train_all[:va_num];t_val = t_train_all[:va_num]
x_train = x_train_all[va_num:];t_train = t_train_all[va_num:]
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
x_test = x_test.astype(np.float32)
train_size = x_train.shape[0]
print(f"Data Split - Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")
iter_per_epoch = max(train_size // batch_size, 1)#48000/256=187.5,48000//256=187니 OK!
# 4. 네트워크 및 옵티마이저 초기화
network = FlexConvNet(input_dim=(1, 28, 28), 
                      conv_param_list=conv_params, 
                      hidden_size_list=hidden_size_list, 
                      output_size=10,
                      weight_init_std='he',
                      use_batchnorm=batchnorm, 
                      use_dropout=True,
                      conv_dropout_ratio=conv_dropout_ratio,
                      fc_dropout_ratio=fc_dropout_ratio,
                      weight_decay_lambda=net_weight_decay)
# 옵티마이저를 먼저 생성 (로드하기 위해 객체가 필요함)
if optimizer_type.lower() == 'adamw':
    optimizer = FlexAdamW(lr=learning_rate, weight_decay=opt_weight_decay)
else:
    optimizer = Adam(lr=learning_rate)
# 학습 상태 변수 및 체크포인트 로드
start_epoch = 0
current_base_lr = learning_rate
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
lr_history = []
best_val_acc = 0
patience_counter = 0
# 기록용 변수 및 Early Stopping 설정
best_params = None
patience = 12#Deep이니 5->7
total_iters = max_epochs * iter_per_epoch
#Loss 폭주 대응과 정체 카운트
prev_epoch_loss = float('inf') # 이전 에폭 Loss 기록
loss_stagnant_cnt = 0          # 정체 카운트
checkpoint_path = "flex_checkpoint.pkl"

# 만약 체크포인트 파일이 존재하면 로드
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # 1. 네트워크 가중치 복구
    network.params = checkpoint['network_params']
    # 레이어 객체와 params를 동기화하는 메서드 호출 (중요)
    network.load_params_from_dict(network.params) 
    
    # 2. 옵티마이저 상태 복구 (Momentum, Velocity)
    optimizer.m = checkpoint['optimizer_m']
    optimizer.v = checkpoint['optimizer_v']
    optimizer.iter = checkpoint['optimizer_iter']
    
    # 3. 학습 카운터 및 히스토리 복구
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    patience_counter = checkpoint['patience_counter']
    current_base_lr = checkpoint['current_base_lr']
    #이전 세션의 최고 가중치 기록
    if 'best_params' in checkpoint:
        best_params = checkpoint['best_params']
    # 그래프를 이어서 그리기 위해 리스트 복구
    train_acc_list = checkpoint['train_acc_list']
    val_acc_list = checkpoint['val_acc_list']
    train_loss_list = checkpoint['train_loss_list']
    val_loss_list = checkpoint['val_loss_list']
    lr_history = checkpoint['lr_history']
    if 'prev_epoch_loss' in checkpoint:
        prev_epoch_loss = checkpoint['prev_epoch_loss']
        loss_stagnant_cnt = checkpoint['loss_stagnant_cnt']
    
    print(f"==> 체크포인트를 발견했습니다! Epoch {start_epoch}부터 학습을 재개합니다.(정체 카운트: {loss_stagnant_cnt}")
#하이퍼파라미터 요약 출력

current_config = {#code 시작시 Hyperparameter 요약용
    'optimizer': optimizer_type,
    'lr': current_base_lr,
    'batch_size': batch_size,
    'max_epochs': max_epochs,
    'max_iterations': iter_per_epoch * max_epochs,
    'conv_param_list': conv_params,
    'hidden_size_list': hidden_size_list,
    'activation': 'relu',          
    'weight_init_std': 'he',        
    'weight_decay_lambda': weight_decay,
    'use_batchnorm': False,
    'conv_dropout_ratio': conv_dropout_ratio,
    'fc_dropout_ratio': fc_dropout_ratio,
}
print("\n" + "="*50)
print("                 [ HYPERPARAMETERS ]")
print(summarize_results(current_config))
print("="*50 + "\n")


# 6. 학습 루프 시작
for epoch in range(start_epoch, max_epochs):
    epoch_start_time = time.time()
    print(f"\n--- Epoch {epoch+1:02d} / {max_epochs:02d} ---")
    #에폭 시작 시 인덱스를 무작위로 한 번 섞어 비복원 추출 준비
    indices = np.arange(train_size);np.random.shuffle(indices)
    epoch_batch_losses = []#get_loss_in_batches, Loss 폭주 대응과 정체 카운트용
    # --- [ Training Phase ] ---
    for i in range(iter_per_epoch):
        #중복 없이 차례대로 배치 사이즈만큼 잘라옴
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_mask = indices[start_idx:end_idx]
        x_batch = x_train[batch_mask];t_batch = t_train[batch_mask]
        """
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        """
        #lr을 초반에는 크게, 후반으로 갈수록 작게 설정
        current_total_iter = (epoch * iter_per_epoch) + i
        new_lr = get_warmup_cosine_lr(current_iter=current_total_iter,total_iters=total_iters,base_lr=current_base_lr,warmup_iters=1 * iter_per_epoch,min_lr=min_lr)
        optimizer.lr = new_lr # 옵티마이저의 학습률을 직접 수정

        # 오차역전파로 기울기 계산
        #gradient에서 loss가 계산되기에 gradient후 다시 loss시에 Conv를 또하는 것임 따라서 gradient에서 loss값을 받아옴
        grads, loss = network.gradient(x_batch, t_batch)

        #lr전략1) Loss Spike 감지 (학습 도중 갑자기 튀는 경우)
        if len(epoch_batch_losses) > 10: # 초반 10번은 통계 확보를 위해 대기
            avg_recent_loss = np.mean(epoch_batch_losses[-10:])
            if loss > avg_recent_loss * 2.0: # 최근 평균보다 2배 이상 튀면
                current_base_lr = max(current_base_lr * 0.8, min_lr) # 전체 기준점 하향
                optimizer.lr *= 0.5 # 현재 보폭 즉시 반토막
                print(f"  [Spike Brake] Iter {i}: Loss {loss:.4f} 튀어오름! LR 긴급 제어")

        # https://dhhwang89.tistory.com/90 https://nmarkou.blogspot.com/2017/07/deep-learning-why-you-should-use.html
        # Gradient Clipping: 기울기 폭주 방지, 비선형 함수에서 미분 값이 매우 크거나 작아지면 마치 가파른 언덕임, 이 결과는 여러개의 큰 가중치값을 곱할때 생기며 이에 다다르면 역전파시에 파라미터들이 크게 움직일 수 있으며 학습을 망침
        #모든 가중치(W,b)의 기울의 L2 Norm을 계산해 너무 크면 방향은 유지하고 길이를 1로 줄임
        #max_norm_in_iter = 0
        clip_value = 1.0#기울기가 1이면 
        for key in grads.keys():#모든 기울기(W1, W2, b1...)를 하나씩 검사
            norm = np.linalg.norm(grads[key])#기울기의 총합인 '길이(L2 Norm)'를 계산
            # if norm > max_norm_in_iter:
            #     max_norm_in_iter = norm  # 가장 큰 가중치 노름 기록
            if norm > clip_value:#만약 기울기가 너무 길다면(강하다면)
                #방향은 유지하되, 길이를 딱 1.0으로 줄여버림
                grads[key] *= (clip_value / norm)
        # if max_norm_in_iter > clip_value:
        #     print(f" [Clip] Exploding gradient capped. Max Norm: {max_norm_in_iter:.2f} -> {clip_value}")
        # 가중치 업데이트
        optimizer.update(network.params, grads)
        
        # 기록: 현재 학습률,loss
        epoch_batch_losses.append(loss)
        if hasattr(optimizer, 'lr_history'):
            lr_history.append(optimizer.lr_history[-1])
        else:
            lr_history.append(optimizer.lr)
        if (i + 1) % 50 == 0 or (i + 1) == iter_per_epoch:
            print(f"  [Iter {i+1:03d}/{iter_per_epoch}] Current Batch Loss: {loss:.4f}")
    # [  에폭 종료 후 정체 판정 및 조치  ]
    # lr전략2) Loss 정체 기반 동적 LR 제어
    avg_train_loss = np.mean(epoch_batch_losses)
    
    # 기준: 0.005 이상 줄어들지 않으면 정체로 판단 (Fashion-MNIST 기준)
    if avg_train_loss >= prev_epoch_loss - 0.005:
        loss_stagnant_cnt += 1
    else:
        loss_stagnant_cnt = 0 # 잘 내려가면 초기화
    # [핵심 추가] 3에폭 동안 정체 시 실제 조치
    if loss_stagnant_cnt >= 3:
        current_base_lr = max(current_base_lr * 0.5, min_lr)
        print(f"  [Dynamic Scheduler] Loss 정체 {loss_stagnant_cnt}회 발생! Base LR 50% 삭감 → {current_base_lr:.6f}")
        loss_stagnant_cnt = 0 # 조치 후 초기화
    prev_epoch_loss = avg_train_loss

    # --- [ Evaluation Phase ] ---
    # 속도를 위한 샘플링 평가
    train_acc = network.accuracy(x_train[:1000], t_train[:1000]) 
    val_acc = network.accuracy(x_val, t_val)
    avg_train_loss = np.mean(epoch_batch_losses)
    val_loss = get_loss_in_batches(network, x_val, t_val, batch_size=batch_size)
    
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(val_loss)
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {avg_train_loss:.4f} | "
          f"TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | "
          f"LR: {optimizer.lr:.6f} | Time: {epoch_duration:.2f}s")
    
    # --- [ Scheduler & Early Stopping 로직 ] ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = pickle.dumps(network.params) # 최고 성능 파라미터 복사 보관
        network.save_params("best_flex_model.pkl") # 파일로도 저장
        patience_counter = 0
    else:
        patience_counter += 1
        # 3번 연속 개선 없으면 학습률 감소
        if patience_counter == 3:
            current_base_lr = max(current_base_lr * 0.5, min_lr)
            print(f"  [Scheduler] Reduce Base LR for Cosine → {current_base_lr}")
        # 5번 연속 개선 없으면 조기 종료
        if patience_counter >= patience:
            print(f"  [Early Stopping] No improvement for {patience} epochs. Training terminated.")
            break
    # --- [ 에폭 루프 끝자락에 추가 ] ---
    # 체크포인트 저장 (모든 동적 변수 포함)
    checkpoint_data = {
        'epoch': epoch, 'network_params': network.params, 'best_params': best_params,
        'optimizer_m': optimizer.m, 'optimizer_v': optimizer.v, 'optimizer_iter': optimizer.iter,
        'best_val_acc': best_val_acc, 'patience_counter': patience_counter, 'current_base_lr': current_base_lr,
        'prev_epoch_loss': prev_epoch_loss, 'loss_stagnant_cnt': loss_stagnant_cnt,
        'train_acc_list': train_acc_list, 'val_acc_list': val_acc_list,
        'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list, 'lr_history': lr_history
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
# 7. 학습 종료 및 최종 평가
print("\n================ Training Finished ================")
# 최고 성능 시점의 파라미터로 복구
if best_params is not None:
    network.params = pickle.loads(best_params)

final_test_acc = network.accuracy(x_test, t_test)
final_loss = train_loss_list[-1]

print("\n" + "="*40)
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Val Acc (Best): {best_val_acc:.4f}")
print(f"Final Test Acc (Real): {final_test_acc:.4f}")
print("="*40)

#plt.show 아래 쓰면 show 그래프 창 꺼야 end_time가 기록됨
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



log_num_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'lognum.txt')
log_num = _get_and_update_log_num(log_num_path)
# 8. 시각화 (정확도 & 학습률)
plt.figure(figsize=(12, 5))
# 정확도 그래프
plt.subplot(1, 2, 1)
x_epochs = np.arange(1, len(train_acc_list) + 1)
plt.plot(x_epochs, train_acc_list, marker='o', label='Train Acc')
plt.plot(x_epochs, val_acc_list, marker='s', label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# 학습률 변화 그래프 (Iteration 단위)
plt.subplot(1, 2, 2)
plt.plot(lr_history, label='Adaptive LR')
plt.yscale('log') 
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.title("Learning Rate History")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'my_plot({log_num}).png')
#plt.show()

# 9. 최종 매개변수 보관 및 시간 출력
save_name = f"flexconvnet_fashion_best({log_num}).pkl"
network.save_params(save_name)
#경로는 save_name가 정의굄으로써 완성
print(f"Saved best model to: {os.path.abspath(save_name)}")
print("============================================================")
#10. test와 비교
print("\n" + " " * 15 + "[ FINAL PRECISION TEST ]")
print("-" * 60)
# 현재 세션에서 저장된 최적 모델 로드
if os.path.exists(save_name):
    network.load_params(save_name)
    print(f"최적 가중치 로드 완료: {save_name}")
# float64(기본 정밀도) 정확도 계산
print("계산 중: Calculate Accuracy (float64) ... ")
acc_f64 = network.accuracy(x_test, t_test)
print(f"결과: Accuracy (float64) = {acc_f64:.4f}")
# float16(반정밀도) 변환 및 정확도 계산
print("계산 중: Calculate Accuracy (float16) ... ")
# 테스트 데이터 형변환
x_test_f16 = x_test.astype(np.float16)
# 네트워크 파라미터 형변환 (현재 network 객체의 params를 직접 수정)
for key in network.params.keys():
    network.params[key] = network.params[key].astype(np.float16)
# 변환된 가중치를 실제 레이어들에 다시 주입 (동기화)
network.load_params_from_dict(network.params)
acc_f16 = network.accuracy(x_test_f16, t_test)
print(f"결과: Accuracy (float16) = {acc_f16:.4f}")
# 정밀도 손실 분석 요약
diff = acc_f64 - acc_f16
print("-" * 60)
print(f"정밀도 차이(f64 - f16): {diff:.6f}")
if abs(diff) < 0.001:
    print("=> 결론: float16 변환 시 정밀도 손실이 거의 없습니다. (매우 견고함)")
else:
    print("=> 결론: float16 변환 시 약간의 정밀도 손실이 발생했습니다.")
print("-" * 60)
print("학습 및 모든 테스트가 종료되었습니다.")
