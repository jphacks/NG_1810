import math
import csv
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, concatenate, LSTM, Flatten, Dropout, Lambda
from keras.utils import np_utils
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
# import matplotlib.pyplot as plt
import pickle
# import seaborn as sns
from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import copy
from sklearn.externals import joblib
from keras.models import model_from_json
import re
import glob
import wave
import unicodedata
import scipy.signal
import pickle
# import matplotlib.pyplot as plt
import copy
import cv2
import numpy as np
import keras
from keras.models import load_model

# 画像をグレースケールで読み込み<入力!!>
img = cv2.imread('./static/img/nami.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# 離散コサイン変換
class DCT:
    def __init__(self, N):
        self.N = N  # データ数
        # 1次元，2次元離散コサイン変換の基底ベクトルをあらかじめ作っておく
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])
        # Nが大きいとメモリリークを起こすので注意
        # MNISTの28x28程度なら問題ない
        self.phi_2d = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    def dct(self, data):
        """ 1次元離散コサイン変換を行う """
        return self.phi_1d.dot(data)

    def idct(self, c):
        """ 1次元離散コサイン逆変換を行う """
        return np.sum(self.phi_1d.T * c, axis=1)

    def dct2(self, data):
        """ 2次元離散コサイン変換を行う """
        N = self.N
        return np.sum(self.phi_2d.reshape(N * N, N * N) * data.reshape(N * N), axis=1).reshape(N, N)

    def idct2(self, c):
        """ 2次元離散コサイン逆変換を行う """
        N = self.N
        return np.sum((c.reshape(N, N, 1) * self.phi_2d.reshape(N, N, N * N)).reshape(N * N, N * N), axis=0).reshape(N,
                                                                                                                     N)

    def phi(self, k):
        """ 離散コサイン変換(DCT)の基底関数 """
        # DCT-II
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos((k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1))


# 以下の２つ関数はメル周波数と周波数の変換を行う関数
def hz2mel(f):
    # Hzをmelに変換
    return 1127.01048 * np.log(f / 700.0 + 1.0)


def mel2hz(m):
    # melをhzに変換
    return 700.0 * (np.exp(m / 1127.01048) - 1.0)


# メルフィルタバンク関数．
def melFilterBank(fs, nfft, numChannels):
    # メルフィルタバンクを作成
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = int(nfft / 2)
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / nfft
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
    filterbank = np.zeros((numChannels, nmax))
    for c in np.arange(0, numChannels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        c = int(c)
        for i in np.arange(indexstart[c], indexcenter[c]):
            i = int(i)
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in np.arange(indexcenter[c], indexstop[c]):
            i = int(i)
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
    return filterbank, fcenters


# STFT(画像を周波数変換をして，分類器の特徴量を設計)
def stft(data):
    cepsfull = []
    deltacepsfull = []

    # オーバーラップの実行
    for i in range(0, 101):
        dct = []
        # 2次元離散コサイン変換
        x = data[i:i + 50, i:i + 50]
        dct_algo = DCT(50)  # 離散コサイン変換を行うクラスを作成
        c = dct_algo.dct2(x)

        # 渦巻きカウンター
        twist = 1
        # 始点・終点を決定する
        firstwidth = int(c.shape[0] / 2)
        firstheight = int(c.shape[1] / 2)
        x = firstwidth
        y = firstheight - 1

        # 奇数番号移動かを振り分ける
        switch = True

        j = 0
        # 2次元を1次元で表現
        while j < c.shape[0] * c.shape[1]:

            # 奇数モード(1,3,5・・)
            if switch:
                for k in range(twist):
                    if j >= c.shape[0] * c.shape[1]:
                        break
                    dct.insert(1, c[x][y])
                    y += 1
                    j += 1
                for k in range(twist):
                    if j >= c.shape[0] * c.shape[1]:
                        break
                    dct.insert(1, c[x][y])
                    x -= 1
                    j += 1
                switch = False
                twist += 1
            # 偶数モード(2,4,6・・)
            else:
                for k in range(twist):
                    if j >= c.shape[0] * c.shape[1]:
                        break
                    dct.insert(1, c[x][y])
                    y -= 1
                    j += 1
                for k in range(twist):
                    if j >= c.shape[0] * c.shape[1]:
                        break
                    dct.insert(1, c[x][y])
                    x += 1
                    j += 1
                switch = True
                twist += 1
        dct = np.array(dct)
        # 離散コサイン変換(メルケプストラム周波数係数の導出)
        ceps = scipy.fftpack.realtransforms.dct(dct, type=2, norm="ortho", axis=-1)
        ceps = ceps[1:13]
        # Δメルケプストラム周波数係数
        deltaceps = np.diff(ceps)
        deltaceps = np.append(deltaceps, 0)

        # 画像の1フレームのメルケプストラムとΔメルケプストラムの特徴量をリストに格納
        cepsfull.append(ceps.tolist())
        deltacepsfull.append(deltaceps.tolist())
    # 画像全体のメルケプストラムとΔメルケプストラムの特徴量を返す
    return cepsfull, deltacepsfull


# 画像を分類器の入力特徴量に変換をする．
cepsall = stft(gray_img)
# 分類器にかける特徴量のため，np.array化
Xmel = np.array([cepsall[0]])
Xmel2 = np.array([cepsall[1]])
# 保存した学習モデルの構造の読み込み
model = model_from_json(open('nn_model.json').read(), custom_objects={"keras": keras})
# 保存した学習モデルの重みの読み込み
model.load_weights('and.h5')
# オノマトペの意味用法リスト(6用法 : drop<落ちる> flow<流れる> rain<雨> soil<土> wave<波> window<風>)
listusage = ["drop", "flow", "rain", "soil", "wave", "window"]
# 読み込んだモデルで予測を出す．
predict_nn = model.predict([Xmel_test, Xmel2_test], batch_size=None, verbose=0, steps=None)
# (usage_answerに1つの用法が入る)<出力!!>
usage_answer = listusage[int(np.max(predict_nn))]