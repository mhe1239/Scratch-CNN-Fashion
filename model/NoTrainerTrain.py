# coding: utf-8
import sys, os, time, pickle
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.mnist import load_mnist 
from flexconvnet import FlexConvNet 
from common.optimizer import *
from common.util import shuffle_dataset

def get_cosine_lr(current_iter, total_iters, base_lr, min_lr=1e-6):
    """코사인 그래프에 따라 학습률을 계산하는 함수"""
    cos_inner = (np.pi * current_iter) / total_iters
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(cos_inner))

def summarize_results(params, final_loss=None, final_acc=None):
    """실험 세팅, CNN 구조 및 결과를 정렬된 텍스트로 요약 출력."""

    use_bn = params.get('use_batchnorm', True)
    dropout_ratio = params.get('dropout_ratio', 0.0)  
    activation = params.get('activation', 'relu')
    weight_init = params.get('weight_init_std', 'relu')

    bn_status = 'T' if use_bn else 'F'
    drop_status = 'T' if dropout_ratio > 0 else 'F'

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
        f"Drop: {drop_status}",
        f"Drop R: {dropout_ratio:<5.1f}"
    ]
    setting_str = ", ".join(fields)
    
    # 2. [추가] conv_param_list 가공 로직
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

    # 3. 결과 지표 병합
    if final_loss is not None and final_acc is not None:
        result_str = f" \n>>> [RESULT] | TRAIN LOSS: {final_loss:.6f} | Val ACC: {final_acc:.4f} |"
        return f"{setting_str}\n{conv_str}{result_str}"
    
    return f"{setting_str}\n{conv_str}"

# 로그 파일 경로 설정 및 초기화 ('w' 모드로 열어 기존 로그를 비우고 새로 시작)
log_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("--- Training Start ---\n")
def _print_and_log(msg):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

# 1. 초기 설정 및 시간 측정 시작
start_time = time.time()
_print_and_log("================ Training Started ================")

# 2. 데이터 로드 및 전처리 (CNN이므로 flatten=False)
(x_train_all, t_train_all), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)

# 데이터 셔플 및 검증 데이터(20%) 분리
x_train_all, t_train_all = shuffle_dataset(x_train_all, t_train_all)
va_rate = 0.2 
va_num = int(x_train_all.shape[0] * va_rate)

x_val = x_train_all[:va_num]
t_val = t_train_all[:va_num]
x_train = x_train_all[va_num:]
t_train = t_train_all[va_num:]
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
x_test = x_test.astype(np.float32)
train_size = x_train.shape[0]
_print_and_log(f"Data Split - Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")
# 3. 네트워크 구조 및 하이퍼파라미터 설정
conv_params = [
    # 블록 1: 28x28 -> 14x14 (채널 32)
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    
    # 블록 2: 14x14 -> 7x7 (채널 64)
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
    
    # 블록 3: 7x7 -> 5x5 (채널 128)
    {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
    {'filter_num':128, 'filter_size':3, 'pad':0, 'stride':1, 'pool':False}# # pad=0이 되면서 (7+0-3)/1 + 1 = 5x5로 축소
    #or {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}# 7x7 이미지 풀링 -> 3x3
]

hidden_size_list = [256, 128] 
learning_rate = 0.001
batch_size = 256  # 두 번째 코드의 고성능 기준 채택
max_epochs = 25   # 첫 번째 코드의 충분한 학습 에폭 채택
dropout_ratio = 0.3
optimizer_type = 'AdamW'
weight_decay = 1e-4
min_lr = 1e-6
# Optimizer 가중치 감쇠(Weight Decay) 처리 분기
if optimizer_type.lower() == 'adamw':
    net_weight_decay = 0
    opt_weight_decay = weight_decay
    
    _print_and_log(f"==> AdamW 감지: Optimizer에서 Weight Decay({weight_decay})를 처리합니다.")
else:
    net_weight_decay = weight_decay
    opt_weight_decay = 0
    _print_and_log(f"==> {optimizer_type} 감지: Network에서 L2 Regularization({weight_decay})을 처리합니다.")

iter_per_epoch = max(train_size // batch_size, 1)

# 4. 네트워크 및 옵티마이저 초기화
network = FlexConvNet(input_dim=(1, 28, 28), 
                      conv_param_list=conv_params, 
                      hidden_size_list=hidden_size_list, 
                      output_size=10,
                      weight_init_std='he',
                      use_batchnorm=True, 
                      use_dropout=True,
                      dropout_ratio=dropout_ratio,
                      weight_decay_lambda=net_weight_decay)

current_config = {
    'optimizer': optimizer_type,
    'lr': learning_rate,
    'batch_size': batch_size,
    'max_epochs': max_epochs,
    'max_iterations': iter_per_epoch * max_epochs,
    'conv_param_list': conv_params,
    'hidden_size_list': hidden_size_list,
    'activation': 'relu',          
    'weight_init_std': 'he',        
    'weight_decay_lambda': weight_decay,
    'use_batchnorm': True,
    'dropout_ratio': dropout_ratio
}

# 시작할 때 설정 상태를 화면과 log.txt에 먼저 띄우기
_print_and_log("\n" + "="*50)
_print_and_log("                 [ HYPERPARAMETERS ]")
_print_and_log(summarize_results(current_config))
_print_and_log("="*50 + "\n")


# FlexAdamW 초기화 (설정 유기적 연결)
if optimizer_type.lower() == 'adamw':
    optimizer = FlexAdamW(lr=learning_rate, weight_decay=opt_weight_decay)
else:
    optimizer = Adam(lr=learning_rate) # 예시용 기본 Adam 혹은 대안

# 5. 기록용 변수 및 Early Stopping 설정
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
lr_history = []

best_val_acc = 0
best_params = None
patience = 7#Deep이니 5->7
patience_counter = 0
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
total_iters = max_epochs * iter_per_epoch
current_base_lr = learning_rate
# 6. 학습 루프 시작
for epoch in range(max_epochs):
    epoch_start_time = time.time()
    batch_loss = []
    log_buffer = []
    _print_and_log(f"\n--- Epoch {epoch+1:02d} / {max_epochs:02d} ---")
    #에폭 시작 시 인덱스를 무작위로 한 번 섞어 비복원 추출 준비
    indices = np.arange(train_size);np.random.shuffle(indices)
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
        new_lr = get_cosine_lr(current_total_iter, total_iters, current_base_lr, min_lr)
        optimizer.lr = new_lr # 옵티마이저의 학습률을 직접 수정

        # 오차역전파로 기울기 계산
        #gradient에서 loss가 계산되기에 gradient후 다시 loss시에 Conv를 또하는 것임 따라서 gradient에서 loss값을 받아옴
        grads,loss = network.gradient(x_batch, t_batch)
        batch_loss.append(loss)
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
        #     _print_and_log(f" [Clip] Exploding gradient capped. Max Norm: {max_norm_in_iter:.2f} -> {clip_value}")
        # 가중치 업데이트
        optimizer.update(network.params, grads)
        
        # 현재 학습률 기록
        if hasattr(optimizer, 'lr_history'):
            lr_history.append(optimizer.lr_history[-1])
        else:
            lr_history.append(optimizer.lr)
        if (i + 1) % 50 == 0 or (i + 1) == iter_per_epoch:
            print(f"  [Iter {i+1:03d}/{iter_per_epoch}] Current Batch Loss: {loss:.4f}")
            log_buffer.append(f"  [Iter {i+1:03d}/{iter_per_epoch}] Current Batch Loss: {loss:.4f}")
    if log_buffer:#2for마다 print하면 아무래도 느려지기에 1for마다 log에 출력
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(log_buffer) + "\n")
    # --- [ Evaluation Phase ] ---
    train_acc = network.accuracy(x_train[:1000], t_train[:1000]) # 속도를 위한 샘플링 평가
    val_acc = network.accuracy(x_val, t_val)
    avg_train_loss = np.mean(batch_loss)
    val_loss = get_loss_in_batches(network, x_val, t_val, batch_size=batch_size)
    
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(val_loss)
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    _print_and_log(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {avg_train_loss:.4f} | "
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
            _print_and_log(f"  [Scheduler] Reduce Base LR for Cosine → {current_base_lr}")
        # 5번 연속 개선 없으면 조기 종료
        if patience_counter >= patience:
            _print_and_log(f"  [Early Stopping] No improvement for {patience} epochs. Training terminated.")
            break
# 7. 학습 종료 및 최종 평가
_print_and_log("\n================ Training Finished ================")
# 최고 성능 시점의 파라미터로 복구
if best_params is not None:
    network.params = pickle.loads(best_params)

final_test_acc = network.accuracy(x_test, t_test)
final_loss = train_loss_list[-1]

_print_and_log("\n" + "="*40)
_print_and_log(f"Final Loss: {final_loss:.4f}")
_print_and_log(f"Final Val Acc (Best): {best_val_acc:.4f}")
_print_and_log(f"Final Test Acc (Real): {final_test_acc:.4f}")
_print_and_log("="*40)

#plt 아래 쓰면 그래프 창 끌때가 end_time이 됨
end_time = time.time()
elapsed_seconds = end_time - start_time
td = timedelta(seconds=elapsed_seconds)
time_str = str(td)

if '.' in time_str:
    time_str = time_str[:time_str.find('.') + 4]
else:
    time_str += '.000'

_print_and_log("============================================================")
_print_and_log(f"Total Elapsed Time: {time_str}")

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
plt.show()

# 9. 최종 매개변수 보관 및 시간 출력
save_name = "flexconvnet_fashion_best.pkl"
network.save_params(save_name)
#경로는 save_name가 정의굄으로써 완성
_print_and_log(f"Saved best model to: {os.path.abspath(save_name)}")
_print_and_log("============================================================")