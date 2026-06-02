"""
BatchNormalization 수정: 기존 common.layers의 BN은 Affine용(2dim)이기에 이미지의 H,W를 구분하지 못해 채널 수와 전체 크기를 맞추지 못함
BatchNormalization 수정: 새로 CNN용으로 만들었으나 numerical_gradient은 노이즈가 생기듯이 BatchNormalization도 오차가 계속 발생하는 구조라서 사용하지 않기로함
-원인: NumPy의 float64 환경에서도 np.sqrt(var + 10e-7)은 매우 미세한 노이즈로써 문제가 생김
문제가 해결됨!!!!!!!!!!
Dropout의 형식을 pytorch에서 쓰는 방식처럼 바꿈 '학습과 테스트 사이의 신호 세기(기댓값)를 맞춘다'라는 수학적 로직은 같지만 기존 방식은 테스트때마다 수백만 개의 뉴런에 스케일링 값을 곱해야 하는 '추론 시 오버헤드'를 가졌다면 Inverted Dropout은 학습 시에 미리 1/(1-p)를 곱해 에너지를 증폭시켜 둠으로써 테스트 단계를 '연산량 0(Identity)'의 상태로 만든다
연산 비용을 완전히 제거한 이 방식은 대규모 시스템에서 압도적인 자원 절약 효과를 가져온다
"""
# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x,train_flg=False):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x,train_flg=False):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x,train_flg=False):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t,train_flg=False):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

#test시에 더 빠른 속도로 연산, 이전의 형식의 pkl가 와도 OK임
class Dropout:
    #Inverted Dropout
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # [Inverted] 학습 때 미리 비율로 나누어 에너지를 키워둠
            return x * self.mask / (1.0 - self.dropout_ratio)
        else:
            # [Inverted] 테스트 때는 아무 연산 없이 통과! (속도 향상)
            return x
"""
class Dropout:
    #http://arxiv.org/abs/1207.0580
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
"""


"""
class BatchNormalization:
    #http://arxiv.org/abs/1502.03167
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
"""
#기존 common.layers의 BN은 Affine용(2dim)이기에 이미지의 H,W를 구분하지 못해 채널 수와 전체 크기를 맞추지 못함
#->노이즈로 인해 사용하지않기로함
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None 

        self.running_mean = running_mean
        self.running_var = running_var          
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape # (N, C, H, W) 또는 (N, D) 저장
        if x.ndim != 2:
            N, C, H, W = x.shape
            # CNN의 경우: 채널(C)을 마지막으로 보내고 펼침 -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape) # 다시 원래 형상으로 복원
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-5)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx



class Convolution:
    """
    W: NDArray[float64],
    b: _Array1D[float64],
    stride: int = 1,
    pad: int = 0
    """
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None
    def forward(self, x,train_flg=False):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        # out은 (N × out_h × out_w, FN)구조의 2차원 메트릭스 이를 다음 layer로 넘기기 위해 4차원으로 변환해야함
        # (0:N, 1:H, 2:W, 3:C)을 (0, 3, 1, 2)로 뒤바꿔 (N, FN, out_h, out_w) format으로 만듦
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # 뒤에서 들어온 4차원을 가중치 미분(dW) 계산을 위해 다시 2차원으로 바꿈
        # (0:N, 1:FN, 2:out_h, 3:out_w) 구조를 (N*out_h*out_w, FN)구조의 2차원 형렬로 바꿔 self.col.T와 행렬 곱이 가능하게 함
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        # 2차원 행렬로 구한 dW를 다시 4차원으로 복원, 모델의 실제 가중치 self.W는 (FN, C, FH, FW)의 4차원 구조이기에 (C*FH*FW, FN)구조를 다시 4차원으로 되돌림
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None
    def forward(self, x,train_flg=False):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        arg_max = np.argmax(col, axis=1)
        # 각 윈도우에서 대푯값 하나 뽑음 
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max
        return out
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        # np.argmax로 윈도우에서 Max값(self.arg_max)을 제외하고 나머지 미분값은 0으로 채워 노이즈 없앰
        # 역전파 시 가중치 업데이트가 핵심 특징을 유발한 뉴런들에게만 집중되어 네트워크 전반에 희소성(Sparsity)이 확보
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx