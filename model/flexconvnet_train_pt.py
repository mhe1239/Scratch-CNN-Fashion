r"""
pt파일을 실행하기 위한 인터프리터 파일 경로C:\pytorch_src\my_env\Scripts
->여기서 python.exe선택
$scriptContent = @'
cd "C:\pytorch_src\model\"
python -u flexconvnet_pt.py *>&1 | Out-File -FilePath "C:\Scratch-CNN-Fashion\model\training.log" -Encoding utf8 -Append
'@
$scriptPath = "C:\Scratch-CNN-Fashion\model\bg_run.ps1"
Set-Content -Path $scriptPath -Value $scriptContent -Encoding utf8
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" -WindowStyle Hidden

Get-Content "C:\Scratch-CNN-Fashion\model\training.log" -Wait

taskkill /F /IM python.exe
"""

"""
pip install torch torchvision numpy matplotlib
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os, pickle
from datetime import timedelta

# [유틸리티] NumPy 버전에서 가져온 번호 관리 함수
def _get_and_update_log_num(log_num_path):
    if os.path.exists(log_num_path):
        with open(log_num_path, 'r', encoding='utf-8') as f:
            try:
                current_num = int(f.read().strip())
            except ValueError:
                current_num = 0
    else:
        current_num = 0

    next_num = current_num + 1
    with open(log_num_path, 'w', encoding='utf-8') as f:
        f.write(str(next_num))
    return current_num

# 1. 데이터 로드 및 전처리 (Fashion-MNIST)
def get_data_loaders(batch_size=256, val_ratio=0.2):
    # [핵심] 학습용은 증강(Augmentation) 적용, 테스트용은 원본 유지
    t_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),#반전
        transforms.RandomRotation(10),#10도 돌리기
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))#Norm
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 데이터 다운로드 (서로 다른 transform 적용을 위해 두 번 로드)
    full_train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=t_train)
    full_val_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=t_test)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=t_test)

    indices = list(range(len(full_train_data)))
    np.random.shuffle(indices)
    split = int(len(full_train_data) * val_ratio)
    train_idx, val_idx = indices[split:], indices[:split]

    # Windows 호환성을 위해 num_workers=0 설정
    train_loader = DataLoader(Subset(full_train_data, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(full_val_data, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, len(train_idx)

# 하이퍼파라미터 출력 로직
def summarize_results(params, final_loss=None, final_acc=None):
    use_bn = params.get('use_batchnorm', True)
    fc_drop = params.get('fc_dropout_ratio', 0.0)
    conv_drop = params.get('conv_dropout_ratio', 0.0)
    
    fields = [
        f"Optimizer: {params.get('optimizer', 'AdamW'):<7}",
        f"LR: {params.get('lr', 'N/A'):<7}",
        f"Batch: {params.get('batch_size', 0):>4}",
        f"Epochs: {params.get('max_epochs', 0):>3}",
        f"Depth(FC): {len(params.get('hidden_size_list', [])):>1}",
        f"BN: {'T' if use_bn else 'F'}",
        f"ConvDrop: {conv_drop:<5.1f}",
        f"FcDrop: {fc_drop:<5.1f}"
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
                 use_batchnorm=True, conv_dropout_ratio=0.0, fc_dropout_ratio=0.2):
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
    print("================ Training Started (PyTorch) ================")

    # 3. 실험 설정
    log_num_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'lognum_pt.txt')
    log_num = _get_and_update_log_num(log_num_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_size = get_data_loaders(batch_size=256)

    conv_params = [
        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}
    ]
    hidden_size_list = [256]
    max_epochs = 30
    lr = 0.001

    current_config = {
        'optimizer': 'AdamW', 'lr': lr, 'batch_size': 256, 'max_epochs': max_epochs,
        'conv_param_list': conv_params, 'hidden_size_list': hidden_size_list,
        'use_batchnorm': True, 'conv_dropout_ratio': 0.0, 'fc_dropout_ratio': 0.5
    }

    print("\n" + "="*50 + "\n                 [ HYPERPARAMETERS ]\n" + summarize_results(current_config) + "\n" + "="*50 + "\n")

    model = FlexConvNet(conv_param_list=conv_params, hidden_size_list=hidden_size_list, fc_dropout_ratio=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=max_epochs)

    # 4. 학습 루프
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0
    model_save_path = f'flex_best_model_pt({log_num}).pt'

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            history['lr'].append(scheduler.get_last_lr()[0])

        train_acc = correct / total
        
        # Validation (evaluation mode)
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
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        duration = time.time() - epoch_start
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {running_loss/len(train_loader):.4f} | "
              f"TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | LR: {history['lr'][-1]:.6f} | Time: {duration:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    # 5. 최종 평가 섹션 (NoTrainerTrain-Test.py 스타일)
    print("\n" + "="*40 + "\n" + " " * 10 + "Training Finished" + "\n" + "="*40)
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

    final_test_acc = test_correct / test_total
    print(f"Final Val Acc (Best): {best_val_acc:.4f}")
    print(f"Final Test Acc (Real): {final_test_acc:.4f}")
    
    elapsed = str(timedelta(seconds=time.time()-start_time))
    print(f"Total Elapsed Time: {elapsed[:elapsed.find('.')+4] if '.' in elapsed else elapsed}")
    print(f"Saved best model to: {os.path.abspath(model_save_path)}")

    # 시각화 및 히스토리 저장
    history_path = f"flex_history_pt({log_num}).pkl"
    with open(history_path, 'wb') as f: pickle.dump(history, f)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy History ({log_num})'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], color='green'); plt.title('Learning Rate History'); plt.yscale('log'); plt.grid(True, alpha=0.3)
    plt.savefig(f'my_plot_pt({log_num}).png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    main()