# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    # sigmoid의 부분값
    # y'=y(1-y)이기에 이를 그대로 적용
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    # y는 신경망의 출력, 예측값/t는 실제 값(test data의 target값)
    # y가 1차원이면 각 데이터 하나마다 교차 엔트로피 오차를 구하기 위해 data의 차원을 바꾼다(reshape)
    if y.ndim == 1:
        # ndim 1->2
        #shape y(10,)->(1,10)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    # one_hot이 True이면 t와 z의 size가 같기에 
    if t.size == y.size:
        # one_hot을 되어있으면 그냥 가장 높은 값인 정답 값을 가져옴(array->int)
        t = t.argmax(axis=1)
    # data의 수
    batch_size = y.shape[0]
    # log0은 -inf니까 계산 불가하니 이거 보정용
    # 0~batch_size-1까지 int 배열 생성하고 그에 해당하는 t의 index위치의 값을 가져옴
    # /batch_size을나누어서 평균 오차를 구함
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)