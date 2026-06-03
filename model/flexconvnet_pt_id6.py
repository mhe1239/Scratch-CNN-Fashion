r"""
pt파일을 실행하기 위한 인터프리터 파일 경로C:\pytorch_src\my_env\Scripts
->여기서 python.exe선택
$scriptContent = @'
cd "C:\Scratch-CNN-Fashion\model"
python -u flexconvnet_pt_id6.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\log\training.log" -Encoding utf8 -Append
'@
$scriptPath = "C:\Scratch-CNN-Fashion\model\bg_run.ps1"
Set-Content -Path $scriptPath -Value $scriptContent -Encoding utf8
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" -WindowStyle Hidden

Get-Content "C:\Scratch-CNN-Fashion\log\training.log" -Wait

taskkill /F /IM python.exe
"""
"""
.pt파일을 기준으로 하도록 설정

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time, sys, os, pickle
from datetime import timedelta
torch.cuda.empty_cache()
def _get_current_log_num(log_num_path):
    """현재 번호가 몇 번인지 읽어오기만 함 (업데이트 X)"""
    if os.path.exists(log_num_path):
        with open(log_num_path, 'r', encoding='utf-8') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0
def _increment_log_num(log_num_path):
    """모든 과정이 성공했을 때만 실행하여 번호를 1 증가시킴"""
    current_num = _get_current_log_num(log_num_path)
    next_num = current_num + 1
    with open(log_num_path, 'w', encoding='utf-8') as f:
        f.write(str(next_num))
    print(f"Log ID {current_num} finalized and updated to {next_num}.")
    return current_num

# 1. 데이터 로드 및 전처리 (Fashion-MNIST)
def get_data_loaders(batch_size=256, val_ratio=0.2):
    # 학습용: 강력한 증강
    t_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),# 평행 이동
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # 검증/테스트용: 원본 유지
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=t_train)
    full_val_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=t_test)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=t_test)
    # 기존에는 train전체를 셔플 후 val를 나눴음: 수정, 셔플하지 않고 그냥 딱 잘라서 나눔
    num_train_all = len(full_train_data)
    num_val = int(num_train_all * val_ratio)
    num_train = num_train_all - num_val
    indices = np.arange(num_train_all)
    np.random.seed(42) 
    np.random.shuffle(indices)
    indices = indices.tolist() # 안전하게 파이썬 리스트로 변환
    # 섞인 인덱스를 기준으로 8:2 영구 고정 분할
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    # 데이터 로더 생성 (훈련셋만 shuffle=True, 검증/테스트는 False)
    train_loader = DataLoader(Subset(full_train_data, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_val_data, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, len(train_idx)
# 하이퍼파라미터 요약 로직
def summarize_results(params):
    use_bn = params.get('use_batchnorm', True)
    fc_drop = params.get('fc_dropout_ratio', 0.0)
    conv_drop = params.get('conv_dropout_ratio', 0.0)
    
    fields = [
        f"Optimizer: {params.get('optimizer', 'AdamW'):<7}",
        f"Peak LR: {params.get('lr', 'N/A'):<7}",
        f"Batch: {params.get('batch_size', 0):>4}",
        f"Epochs: {params.get('max_epochs', 0):>3}",
        f"Depth(FC): {len(params.get('hidden_size_list', [])):>1}",
        f"BN: {'T' if use_bn else 'F'}",
        f"ConvDrop: {conv_drop:<5.1f}",
        f"FcDrop: {fc_drop:<5.1f}",
        f"WD: {params.get('weight_decay', 0):<7}"
    ]
    setting_str = ", ".join(fields)
    
    conv_lines = ["\n  [ CNN Architecture Flow (PyTorch) ]", "  " + "-" * 55, 
                  "  | Layer | Filter Num | Filter Size | Pad | Stride | Pool  |", "  " + "-" * 55]
    for i, c in enumerate(params.get('conv_param_list', [])):
        conv_lines.append(f"  | C_{i+1:<3} | {c['filter_num']:^10} | {c['filter_size']:^11} | {c['pad']:^3} | {c['stride']:^6} | {'T' if c.get('pool') else 'F':^5} |")
    conv_lines.append("  " + "-" * 55)
    return f"{setting_str}\n" + "\n".join(conv_lines)

# 2. PyTorch용 FlexConvNet 클래스
class FlexConvNet(nn.Module):
    def __init__(self, input_dim=(1, 28, 28), conv_param_list=[], hidden_size_list=[128], output_size=10, 
                 use_batchnorm=True, conv_dropout_ratio=0.0, fc_dropout_ratio=0.5):
        super(FlexConvNet, self).__init__()
        layers = []
        in_channels = input_dim[0]
        curr_h, curr_w = input_dim[1], input_dim[2]
        for prec in conv_param_list:
            layers.append(nn.Conv2d(in_channels, prec['filter_num'], prec['filter_size'], prec['stride'], prec['pad']))
            if use_batchnorm: layers.append(nn.BatchNorm2d(prec['filter_num']))
            layers.append(nn.ReLU())
            if prec.get('pool', False):
                layers.append(nn.MaxPool2d(2, 2))
                curr_h //= 2; curr_w //= 2
            if conv_dropout_ratio > 0: layers.append(nn.Dropout2d(conv_dropout_ratio))
            in_channels = prec['filter_num']
        self.features = nn.Sequential(*layers)
        self.flat_size = in_channels * curr_h * curr_w
        fc_layers = []
        in_features = self.flat_size
        for h_size in hidden_size_list:
            fc_layers.append(nn.Linear(in_features, h_size))
            if use_batchnorm: fc_layers.append(nn.BatchNorm1d(h_size))
            fc_layers.append(nn.ReLU())
            if fc_dropout_ratio > 0: fc_layers.append(nn.Dropout(fc_dropout_ratio))
            in_features = h_size
        fc_layers.append(nn.Linear(in_features, output_size))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    start_time = time.time()
    
    
    print(f"================ Training Started (PyTorch) ================")
    print(f"{os.path.basename(__file__)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_size = get_data_loaders(batch_size=256)
    
    conv_params = [
        {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
        {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},  # 14x14
        {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
        {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},  # 7x7
        {'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1, 'pool':False},
        {'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}   # 3x3
    ]
    max_epochs = 40
    lr = 0.00005
    wd = 0.1
    patience=25
    hidden_size_list=[512]
    config = {
        'optimizer': 'AdamW', 'lr': lr, 'batch_size': 256, 'max_epochs': max_epochs,
        'conv_param_list': conv_params, 'hidden_size_list': hidden_size_list,
        'use_batchnorm': True, 'conv_dropout_ratio': 0.0, 'fc_dropout_ratio': 0.5, 'weight_decay': wd,'label_smoothing': 0.1
    }
    print("="*50 + "\n                 [ HYPERPARAMETERS ]\n" + summarize_results(config) + "\n" + "="*50 + "\n")

    model = FlexConvNet(conv_param_list=conv_params, hidden_size_list=hidden_size_list, fc_dropout_ratio=0.5).to(device)
    prev_model_path = 'flex_best_model_pt(5).pt'
    if os.path.exists(prev_model_path):
        model.load_state_dict(torch.load(prev_model_path))
        print(f"==> [{prev_model_path}] 로드 완료! 96%를 향한 Fine-tuning을 시작합니다.")
    else:
        print(f" [{prev_model_path}]를 찾을 수 없습니다. 처음부터 학습합니다.")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    max_epochs = 40
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)#모델이 정답을 100% 확신하지 못하게 살짝 깎아주는 기법
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc, prev_epoch_loss, loss_stagnant_cnt = 0, float('inf'), 0
    log_num_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'lognum_pt.txt')
    log_num =_get_current_log_num(log_num_path)
    peak_reached, model_save_path = False, f'flex_best_model_pt({log_num}).pt'

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_batch_losses = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # # [모니터링 1] Peak LR 알림
            # current_lr = scheduler.get_last_lr()[0]
            # if not peak_reached and current_lr >= lr * 0.95:
            #     print(f"  [ Scheduler] Peak LR ({current_lr:.6f}) reached! Starting Annealing...")
            #     peak_reached = True
            
            # [모니터링 2] Spike 감지
            if len(epoch_batch_losses) > 10:
                if loss.item() > np.mean(epoch_batch_losses[-10:]) * 2.0:
                    print(f"  [ Spike detected] Iter {i}: Loss {loss.item():.4f} 튀어오름!")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()
            
            running_loss += loss.item()
            epoch_batch_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # history['lr'].append(current_lr)

        # [수정] 에폭이 끝날 때 학습률 기록 및 스케줄러 업데이트
        current_lr = scheduler.get_last_lr()[0]
        history['lr'].append(current_lr)
        scheduler.step() # 여기서 에폭 단위로 학습률을 조금씩 깎음

        train_acc = correct / total
        avg_train_loss = running_loss / len(train_loader)
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {avg_train_loss:.4f} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {history['lr'][-1]:.6f} | Time: {time.time()-epoch_start:.2f}s")

        if val_acc > best_val_acc:
            print(f"  [Best] New Record! Val Acc: {val_acc:.4f} (Previous: {best_val_acc:.4f})")
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:
                print(f"  [Early Stopping] No improvement for {patience_counter} epochs.")
        if patience_counter >= patience:
            print(f"\n[ Early Stopping] {patience} 에폭 동안 개선이 없어 학습을 종료합니다.")
            break
    _increment_log_num(log_num_path)

    print("\n" + "="*40 + "\n          Training Finished\n" + "="*40)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print(f"Final Val Acc (Best): {best_val_acc:.4f}")
    print(f"Final Test Acc (Real): {test_correct / test_total:.4f}")
    elapsed = str(timedelta(seconds=time.time()-start_time))
    print(f"Total Elapsed Time: {elapsed[:elapsed.find('.')+4] if '.' in elapsed else elapsed}"+'\n'+"="*40)
    with open(f"flex_history_pt({log_num}).pkl", 'wb') as f: pickle.dump(history, f)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy History ({log_num})'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2); plt.plot(history['lr'], color='green'); plt.title('Learning Rate History'); plt.yscale('log'); plt.grid(True, alpha=0.3)
    plt.savefig(f'my_plot_pt({log_num}).png', bbox_inches='tight')
    
    plt.close('all')
    if torch.cuda.is_available():#GPU 캐시 지우기
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
    print("Learning and all tests have been completed.")
    sys.exit(0)