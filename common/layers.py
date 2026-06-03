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
# 원인: Scratch로 구현된 BatchNorm에서 /로 나눈 경우 부동 소수점을 모두 저장 할 수 없음(pi의 무한 소수를 담는 다면 float64까지가 최대이고 이후부터는 짤림) 이에 따라 오차가 발생함 이를 위해 __forward의 eps을 1e-7이 아닌 1e-5으로 바꿈 그리고 var와 std는 수학적으론 0이상이 나오지만 부동소수점 문제로 미세하게 작은 -값이 나올 수 있기에 np.maximum을 적용함
# 또한 /연산이 많은 __backward을 간결하게 바꿈 
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
        eps = 1e-5 # 오차 상향 조정
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.maximum(np.mean(xc**2, axis=0), 0)#var = np.mean(xc**2, axis=0), 수학적으로 분산은 음수가 나올 수 없지만 x-mu과정에서 부동소수점 오차로 인해 var가 0보다 미세하게 작은 -1e-18같은 값이 나올 수 있음, np.sqrt은 음수 받으면 NaN을 반환하기에 이를 방지하는 것이 오류 누적을 막음
            self.std = np.sqrt(var + eps)#std = np.sqrt(var + eps)
            xn = xc / self.std#xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            # self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + eps)
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx
    """
    def __backward(self, dout):
        std = np.maximum(self.std, 1e-5)#var와 같은 혹시 모를 std 값의 소실 방지
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / std#위에서 정의한 std사용
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx
    """
    # https://arxiv.org/abs/1502.03167 __backward에서 사용 중인 수식은 사실 BatchNorm의 역전파를 수학적으로 가장 충실하게 구현한 방식이지만 /로 나누는 로직이 많기에 부동소수점 오차(Floating-point drift)가 쌓일 가능성이 있기에 이를 더 안정적이고 간결하게 바꿈
    def __backward(self, dout):
        self.dgamma = np.sum(self.xn * dout, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        safe_std = np.maximum(self.std, 1e-5)#수치적 안정성을 위해 std 값을 하한선으로 고정 (eps와 동일한 1e-5 사용)
        # 2. BatchNorm의 역전파 핵심 (간결화)
        # 공식: dx = (gamma / std) * [dout - mean(dout) - xn * mean(dout * xn)]
        # 이 방식은 중간 변수(dstd, dvar 등)를 많이 거치지 않아 오차 축적을 방지
        N = self.batch_size
        dxn = self.gamma * dout
        # 여기서 mean() 연산을 효율적으로 사용, 이 한 줄이 위에서 쓰던 복잡한 4~5줄의 수식을 대체
        dx = (1.0 / (N * safe_std)) * ( N * dxn - np.sum(dxn, axis=0) - self.xn * np.sum(dxn * self.xn, axis=0))
        return dx


# Scratch로 구현된 Conv는 BatchNrom과는 다른 문제가 있는데 이는 CNN의 깊은 층으로 갈수록 발생하는 '데이터의 분산 소실/폭주'와 관련이 깊음
# 가장 큰 문제, 가중치 정규화(BatchNrom)가 없음: Convolution 레이어는 Affine 레이어보다 가중치 개수(Weight parameters)가 훨씬 많은데 BatchNormalization가 없으면 층이 5-6개만 넘어가도 입력 데이터의 분포가 레이어를 지날수록 0으로 수렴하거나 무한대로 발산한다
# ->Convolution 연산은 가중치를 공유하는 행렬 곱의 연속인데 BatchNrom이 없으면 다음과 같은 문제가 발생함
# - 1. Internal Covariate Shift: Conv 레이어를 통과할 때마다 출력값의 평균과 분산이 계속 변함, 뒤로 갈수록 분포가 극단적으로 쏠리게 됨
# - 2. Activation Saturation: ReLU를 사용 중이면 분포가 음수 쪽으로 너무 많이 쏠릴 경우 모든 뉴런이 0을 출력하는 Dying ReLU 현상이 발생
# - 3. 가중치 크기(Scale): Affine은 입력 차원이 고정적이지만, Conv는 Filter size * Filter size * Channel만큼의 입력을 받음. 이 때문에 He 초기화가 완벽하지 않으면 초반부터 학습이 꼬임
# backward에서의 pad 복원: col2im은 이미지 패딩을 포함한 크기로 되돌리는데, 마지막에 슬라이싱([:, :, pad:H + pad, ...])을 할 때, 만약 stride가 1이 아니라면 col2im 내부의 img 생성 크기가 미세하게 달라질 수 있다
# BatchNorm이 없다면 lr은 1e-3~1e-4로 잡아야하며 lr이 0.01 이상으로 높다면 BatchNorm이 없는 CNN은 수렴하기 어려움
class Convolution:
    """
    W: NDArray[float64],
    b: _Array1D[float64],
    stride: int = 1,
    pad: int = 0
    """
    def __init__(self, W, b, stride=1, pad=0, use_batchnorm=False):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.use_batchnorm = use_batchnorm # BN 사용 여부 저장
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None
    def forward(self, x,train_flg=False):#4차원 행렬을 일반적인 행렬곱(dot)을 위해 y=xw+b형태로 변환
        #가중치 필터(w)에서 필터수,채널수,필터h,필터w 추출
        FN, C, FH, FW = self.W.shape
        # 입력 데이터에서 배치 크기,채널수,높이,너비 추출
        N, C, H, W = x.shape
        #out 크기 구함
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        # 이미지에서 필터가 마주치는 윈도우 영역인 3차원 영역(c,fn,fw)를 모두 1차원으로 펼쳐 2차원 행렬 만듦
        col = im2col(x, FH, FW, self.stride, self.pad)
        # #4차원 필터도 하나당 1개의 열을 차지하도록 1차원으로 변형 후 곱을 위해 전치(.T)시킨다
        col_W = self.W.reshape(FN, -1).T
        # 변형된 두 행렬을 np.dot으로 한 번에 내적 연산하고, 편향 b를 더함
        out = np.dot(col, col_W) + self.b
        # out은 (N × out_h × out_w, FN)구조의 2차원 메트릭스 이를 다음 layer로 넘기기 위해 4차원으로 변환해야함
        # (0:N, 1:H, 2:W, 3:C)을 (0, 3, 1, 2)로 뒤바꿔 (N, FN, out_h, out_w) format으로 만듦
        if not self.use_batchnorm:# out의 값은 0.1-0.5
            out *= 0.5 # Activation Scaling BN을 쓰지 않는 경우 Conv의 출력을 강제로 일정 수준으로 맞춰주는 고정 스케일링
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        #역전파를 위해 저장
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # 뒤에서 들어온 4차원을 가중치 미분(dW) 계산을 위해 다시 2차원으로 바꿈
        # (0:N, 1:FN, 2:out_h, 3:out_w) 구조를 (N*out_h*out_w, FN)구조의 2차원 형렬로 바꿔 self.col.T와 행렬 곱이 가능하게 함
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        # 변향과 가중치의 기울기를 계산한다
        # dout을 세로축을 모두 더해 편향의 기울기인 db를 구함(b는 필터당 1개씩 존재해서)
        self.db = np.sum(dout, axis=0)
        # 순반향때 전개했던 이미지 행렬인 col.T와 out을 곱해 가중치 기울기 dW를 구함
        self.dW = np.dot(self.col.T, dout)
        # 2차원 행렬로 구한 dW를 다시 4차원으로 복원, 모델의 실제 가중치 self.W는 (FN, C, FH, FW)의 4차원 구조이기에 (C*FH*FW, FN)구조를 다시 4차원으로 되돌림
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        # # dout에 순방향 행렬의 전치인 col_W.T를 곱해 펼쳐진 이미지 형태의 기울기인 dcol(이때 여전히 2차원 행렬 상태)을 계산함
        dcol = np.dot(dout, self.col_W.T)
        # 2차원 형태인 dcol을 입력이미지 모형인 x.shape의 4차원 구조로 복원해 넘김
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
        # 입력 데이터에서 배치 크기,채널수,높이,너비 추출
        N, C, H, W = x.shape
        # out 크기 구함
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 4차원 이미지에서 풀링 윈도우가 지나가는 영역들을 한 줄짜리 1차원 배열로 펼쳐 2차원 행렬로 만듦
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 전개된 행렬을 [전체 윈도우 개수, 하나의 윈도우 안의 원소 개수] 모양의 2차원 행렬로 재정렬해 윈도우별로 Max값을 찾기 쉽게 한다
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # 윈도우 내의 가장 큰 값이 몇번째 열에 있었는지 그 index를 기록
        arg_max = np.argmax(col, axis=1)
        # 각 윈도우에서 대푯값 하나 뽑음 
        out = np.max(col, axis=1)
        # 더 쉬운 연산을 위해 가져옴
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # 역전파를 위해 저장
        self.x = x
        self.arg_max = arg_max
        return out
    def backward(self, dout):
        # forward로 바꿨기에 다시 transpose함 
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        # np.argmax로 윈도우에서 Max값(self.arg_max)을 제외하고 나머지 미분값은 0으로 채워 노이즈 없앰
        # 역전파 시 가중치 업데이트가 핵심 특징을 유발한 뉴런들에게만 집중되어 네트워크 전반에 희소성(Sparsity)이 확보
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        # 2차원 행렬이었던 dmax를 다시 4차원 형태에 가깝게 변형
        dmax = dmax.reshape(dout.shape + (pool_size,))
        # 방금 만든 5차원 행렬을 다시 col2im가 인식 가능한 2차원 행렬로 변환
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # 2차원으로 펼쳤던 기울기 행렬 dcol을 원래 이미지 형상인 4차원인 dx와 똑같게 조립함
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

# CNN에서는 채널 단위 정규화인 GroupNorm가 LayerNorm보다 좋다곤 함
class LayerNormalization:
    def __init__(self, gamma, beta, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.xn = None
        self.std = None
        self.x_shape = None
    def forward(self, x, train_flg=True):#LayerNorm에선 train_flg가 T/F이든 상관없음
        self.x_shape = x.shape
        # 마지막 차원을 기준으로 평균과 분산 계산
        # x: (N, D) 또는 (N, C, H, W)에서 마지막 축을 기준으로 처리
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        self.std = np.sqrt(var + self.eps)
        self.xn = (x - mu) / self.std
        
        return self.gamma * self.xn + self.beta
    def backward(self, dout):
        # LayerNorm의 역전파는 BatchNorm보다 훨씬 깔끔
        # D는 특징 차원(마지막 축의 크기)
        N = dout.shape[-1]
        # 가중치 미분
        self.dgamma = np.sum(dout * self.xn, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        # 입력에 대한 미분 (중간 변수 최소화로 오차 감소)
        dx = (self.gamma / self.std) * ( dout - np.mean(dout, axis=-1, keepdims=True) - self.xn * np.mean(dout * self.xn, axis=-1, keepdims=True))
        return dx

class GroupNormalization:
    #gamma는 np.ones(C), beta는 np.zeros(C)로 초기화하여 사용
    # group=2 or group=4#채널 수 C가 작다면(예: 32) group=8 정도로 설정하여 그룹당 채널 수를 4개 정도로 유지하는 것이 성능이 좋음
    def __init__(self, gamma, beta, group=2, eps=1e-5):
        self.gamma = gamma # (C,)
        self.beta = beta   # (C,)
        self.group = group
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.group
        
        # 1. 데이터를 그룹화: (N, G, C//G, H, W)
        x_reshaped = x.reshape(N, G, C // G, H, W)
        
        # 2. 각 그룹 내에서 평균과 분산 계산
        # axis=(2, 3, 4)는 C//G, H, W를 의미
        mu = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)
        
        # 3. 정규화
        self.std = np.sqrt(var + self.eps)
        self.xn = (x_reshaped - mu) / self.std
        
        # 4. 원래 차원으로 복원 후 감마/베타 적용
        out = self.xn.reshape(N, C, H, W) * self.gamma.reshape(1, C, 1, 1) + self.beta.reshape(1, C, 1, 1)
        return out

    def backward(self, dout):
        N, C, H, W = dout.shape
        G = self.group
        
        # 1. 감마,베타 초기화
        dgamma = np.sum(dout * self.xn.reshape(N, C, H, W), axis=(0, 2, 3))
        dbeta = np.sum(dout, axis=(0, 2, 3))
        # 2. dout을 그룹 형태로 변환하여 역전파 계산
        dout_reshaped = dout.reshape(N, G, C // G, H, W)
        dxn = dout_reshaped * self.gamma.reshape(1, G, C // G, 1, 1)
        
        # 3. GN 역전파 수식 (그룹 단위 통계량 제거)
        # S: 그룹 내 원소 수 (C//G * H * W)
        S = (C // G) * H * W
        
        dmu = -np.sum(dxn / self.std, axis=(2, 3, 4), keepdims=True)
        dvar = -0.5 * np.sum(dxn * (self.xn / self.std), axis=(2, 3, 4), keepdims=True)
        
        dx = (dxn / self.std) + (dvar * 2 * (self.xn * self.std) / S) + (dmu / S)
        
        return dx.reshape(N, C, H, W)