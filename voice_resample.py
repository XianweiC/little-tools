import librosa as lr
import numpy as np


"""
@params array: 原始数组
@return:一个同长度的新数组

"""
def get_index(array):
    new_array = np.zeros(len(array))
    for i in range(len(array)):
        if array[i] != 0:
            new_array[i] = 1
    return new_array


'''@param array:'''
def get_index1(array, flag):
    """
    :param array:原始数组
    :param flag: 要判断的元素
    :return: 一个同长度的新数组
    """
    new_array = np.zeros(len(array))
    for i in range(len(array)):
        if array[i] == flag:
            new_array[i] = 1
    return new_array


# filename = "LEADER.wav"
filename = "爷爷，该吃药了.wav"
# 设置采样频率
sample_rate = 48000
# 以指定的采样频率读取音频文件
Y, Fs = lr.load(filename, sr=sample_rate)

OriginalSign = Y

b = 0.8

# b_rate = [43200, 38400, 33600, 28800]
# ResampleSign = lr.resample(Y, sample_rate, b_rate[i])
# ResampleSign = lr.resample(Y, sample_rate, 43200)  #0.9
# ResampleSign = lr.resample(Y, sample_rate, 38400)  #0.8
# ResampleSign = lr.resample(Y, sample_rate, 33600)  #0.7
ResampleSign = lr.resample(Y, sample_rate, 28800)  #0.6
# ResampleSign = lr.resample(Y, sample_rate, 50000)



S = 400
Overlap = 200
Pmax = 170
B = S - Overlap
OriginalLen = len(Y)
NewSign = np.zeros(OriginalLen)
ratio = np.array([i / (Overlap + 1) for i in range(1, Overlap + 1)])

i = np.array([i for i in range(1, Overlap + 1)])

app = np.array([i for i in range(1, B + 1)])

CalSeries = np.hstack((ResampleSign, np.zeros(400)))

NewSign[1:S] = ResampleSign[1:S]

rag = [i for i in range(B, (OriginalLen - S), B)]
for newpos in rag:
    Originalpos = round(b * newpos)
    index = i + newpos
    y = NewSign[index]
    rxy = np.zeros(Pmax + 1)
    rxx = np.zeros(Pmax + 1)
    Pmin = 0
    for p in range(Pmin, Pmax + 1):
        index1 = Originalpos + p + i
        x = CalSeries[index1]
        rxx[p] = np.linalg.norm(x)
        rxy[p] = np.sum(y * x)
    rxx_ = get_index(rxx)
    rxx__ = get_index1(rxx, 0)
    Rxy = rxx_ * rxy / (rxx + rxx__)
    index2 = get_index1(Rxy, max(Rxy))
    pm = np.min(np.nonzero(index2))
    bestpos = Originalpos + pm

    NewSign[newpos + i] = ((1 - ratio) * NewSign[newpos + i]) + (
        ratio * CalSeries[bestpos + i]
    )
    NewSign[newpos + Overlap + app] = CalSeries[bestpos + Overlap + app]

NewSign = NewSign.astype(np.float32)  # 将数据格式从float64重新转换为float32
lr.output.write_wav("output.wav", NewSign, sample_rate)
