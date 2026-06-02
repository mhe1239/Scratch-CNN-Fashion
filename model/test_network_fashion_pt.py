import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


TEST_MODEL_ID = 4  # 테스트하고 싶은 모델의 ID 번호
MODEL_PATH = f'flex_best_model_pt({TEST_MODEL_ID}).pt'

# 2. 모델 구조 정의(학습때와 같아야함!)
class FlexConvNet(nn.Module):
    def __init__(self, input_dim=(1, 28, 28), conv_param_list=[], hidden_size_list=[128], output_size=10, 
                 use_batchnorm=True, fc_dropout_ratio=0.5):
        super(FlexConvNet, self).__init__()
        layers = []
        in_channels = input_dim[0]
        for prec in conv_param_list:
            layers.append(nn.Conv2d(in_channels, prec['filter_num'], prec['filter_size'], prec['stride'], prec['pad']))
            if use_batchnorm: layers.append(nn.BatchNorm2d(prec['filter_num']))
            layers.append(nn.ReLU())
            if prec.get('pool', False): layers.append(nn.MaxPool2d(2, 2))
            in_channels = prec['filter_num']
        self.features = nn.Sequential(*layers)
        
        # 3층 와이드 구조 기준의 flat_size 계산 (28->14->7->3)
        # 만약 풀링 구조가 다르면 이 부분은 자동으로 계산되게 하거나 맞춰줘야함
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 3 * 3, hidden_size_list[0]),
            nn.ReLU(),
            nn.Dropout(fc_dropout_ratio),
            nn.Linear(hidden_size_list[0], output_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Model ID: {TEST_MODEL_ID} Testing on {device} ---")

    # 3. 테스트 데이터 준비 (학습 때와 동일한 정규화 필수!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # 4. 모델 생성 및 가중치 로드
    conv_params = [
        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True},
        {'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1, 'pool':True}
    ]
    model = FlexConvNet(conv_param_list=conv_params, hidden_size_list=[256]).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"에러: {MODEL_PATH} 파일을 찾을 수 없습니다!")
        return

    # 가중치 파일 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # 추론 모드로 전환 (Dropout, BN 고정)

    # 5. 정확도 계산
    correct = 0
    total = 0
    with torch.no_grad(): # 기울기 계산 끔 (속도/메모리 절약)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    print(f"결과: {MODEL_PATH}의 최종 테스트 정확도 = {accuracy:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_test()