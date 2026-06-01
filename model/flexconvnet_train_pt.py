r"""
$scriptContent = @'
cd "C:\pytorch_src\model\"
python -u flexconvnet_pt.py *>&1 | Out-File -FilePath "C:\src(3-1)\ArtificialIntelligence\w99\training.log" -Encoding utf8 -Append
'@
$scriptPath = "C:\src(3-1)\ArtificialIntelligence\w99\bg_run.ps1"
Set-Content -Path $scriptPath -Value $scriptContent -Encoding utf8
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" -WindowStyle Hidden

Get-Content "C:\src(3-1)\ArtificialIntelligence\w99\training.log" -Wait

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
import os,pickle
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
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),## 좌우 반전 (셔츠는 뒤집어도 셔츠)
         transforms.RandomRotation(10),      # 10도 이내 무작위 회전
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Fashion-MNIST 표준 정규화
    ])

    # 데이터 다운로드
    full_train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Validation Split (NumPy 버전과 동일한 로직)
    num_train = len(full_train_set)
    indices = list(range(num_train))
    split = int(np.floor(val_ratio * num_train))
    
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(full_train_set, train_idx)
    val_subset = Subset(full_train_set, val_idx)

    # Windows 호환성을 위해 num_workers=0 설정
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# 2. PyTorch용 FlexConvNet 클래스 (이전 설계와 동일)
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
    # 3. 학습 설정 루틴
    log_num_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'lognum_pt.txt') # PT 전용 번호 파일
    log_num = _get_and_update_log_num(log_num_path)
    model_save_path = f'flex_best_model_pt({log_num}).pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders()

    conv_params = [
        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}
    ]

    model = FlexConvNet(conv_param_list=conv_params, hidden_size_list=[256],fc_dropout_ratio=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # PyTorch의 강력한 스케줄러: OneCycleLR (Warmup + Cosine을 한 번에 제어)
    max_epochs = 30
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=max_epochs)

    # 4. 학습 루프
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0
    patience, patience_counter = 10, 0
    start_time = time.time()

    print(f"Training on {device}...")

    for epoch in range(max_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # NumPy에서 했던 Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # 이터레이션마다 LR 업데이트 (코사인 스케줄링)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            history['lr'].append(scheduler.get_last_lr()[0])

        train_acc = correct / total
        
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
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Loss: {running_loss/len(train_loader):.4f} | "
            f"TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | LR: {history['lr'][-1]:.6f}")

        # Best Model Save & Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # 5. 최종 평가 및 시각화
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

    print("\n" + "="*40)
    print(f"Final Test Accuracy: {test_correct / test_total:.4f}")
    print(f"Total Time: {str(timedelta(seconds=time.time()-start_time))}")
    print("="*40)
    # 1. 학습 히스토리 저장 (.pkl)
    history_path = f"flex_history_pt({log_num}).pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    # 그래프 그리기
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.title('Learning Rate History (OneCycleLR)')
    plt.savefig(f'my_plot_pt({log_num}).png')
    plt.show()
if __name__ == '__main__':
    main()  