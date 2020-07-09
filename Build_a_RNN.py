import numpy as np
import torch
import torch.nn.functional as F

# in here, a is hidden state.
def rnn_cell_forward(xt, a_prev, parameters):
    # RNN is repetition of the RNN cell.

    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute next activation state using the formula given above
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # compute output of the current cell using the formula given above
    before_softmax = torch.from_numpy(np.dot(Wya, a_next) + by)
    yt_pred = F.softmax(before_softmax).numpy()  # numpy로 다시 저장.

    # store values you need for "backward propagation through time" in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

def rnn_forward_pass(x, a0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    # T_x is # training set
    n_y, n_a = parameters["Wya"].shape

    # initialize
    a = np.zeros((n_a, m, T_x))  # we will stack
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward([x[:, :, t], a_next, parameters])
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred

    # accumulate
    caches.append(cache)
    caches = (caches, x)

    return a, y_pred, caches

def rnn_cell_backward(da_next, cache):  # considering BPTT

    (a_next, a_prev, xt, parameters) = cache
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 수식을 고대로 가져왔네,,,
    # BPTT 를 내가 직접 짜야하나?

    # ★ 근데, 그 전에... np.dot이랑 * 차이점도 간단하게 리뷰해야할 듯.


    # J에 대한 미분식을 얻기 위해 a_next Node를 거쳐 chain rule 적용.

    common = (1 - a_next*a_next)  # 그냥, 반복되는 항.
    # dxt = da_next/dxt * dJ/da_next = (or chanin rule) local * up
    dxt = np.dot(Wax.T*common, da_next)
    dWax = np.dot(common*xt.T, da_next)
    dWaa = np.dot(common*a_prev.T, da_next)
    da_prev = np.dot(Waa.T*common, da_next)
    dba = np.sum(common*da_next, keepdims=True, axis=-1)
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    return gradients

def rnn_backward_pass(da, caches):  # accumulate
    caches, x = caches
    a1, a0, x1, parameters = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((parameters['Wax'].shape))
    dWaa = np.zeros((parameters['Waa'].shape))
    dba = np.zeros((parameters['ba'].shape))
    da0 = np.zeros(a0.shape)
    da_prevt = np.zeros((n_a, m))


    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], \
                                           gradients['da_prev'], \
                                           gradients['dWax'],\
                                           gradients['dWaa'],\
                                           gradients['dba']
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        dx[:, :, t] = dxt
        da0 = da_prevt

    gradietns = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

# LSTM forward, backward 구조 설계는 나중에