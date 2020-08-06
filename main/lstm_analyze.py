#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/16
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import datetime


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def lstm_model(timestamp,predict_size):
    model = keras.Sequential([
    layers.LSTM(20,return_sequences=False,
                input_shape=(timestamp, 1),
                # activation='relu',
                use_bias=True),
    # layers.Dropout(0.2),
    # layers.LSTM(10, return_sequences=False,use_bias=False),
    layers.Dropout(0.3),
    layers.Dense(predict_size),
    ])
    # (20, input_shape=(timestamp, 1), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False)
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss='mean_squared_error',
                 metrics=['accuracy'])
    return model

def lstm_train(df, predict_size,sc_fit,timestamp):
    train = df
    # 归一化
    train_sc = sc_fit.transform(train[:,np.newaxis])

    # 将时序数据转为输入长度为timestamp，预测长度为predict_size的数组，等于说其shape为[train_sc.shape[0],timestamp,predict_size]
    rt = series_to_supervised(train_sc,timestamp,predict_size)
    reframed_train  = series_to_supervised(train_sc,timestamp,predict_size).values

    train_X, train_y = reframed_train[:, :-predict_size], reframed_train[:, -predict_size:]

    X_tr_t = train_X.reshape((train_X.shape[0], timestamp,1))

    # 销毁当前的TF图并创建一个新图。
    # 有助于避免旧模型 / 图层混乱。
    # tf.keras.backend.clear_session()
    model_lstm = lstm_model(timestamp,predict_size)

    # verbose = 0，在控制台没有任何输出
    # verbose = 1 ：显示进度条
    # verbose =2：为每个epoch输出一行记录
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
    history_model_lstm = model_lstm.fit(X_tr_t, train_y, epochs=10, batch_size=2, verbose=0, shuffle=False,
                                        callbacks=[early_stop])
    return history_model_lstm

def lstm_predict(model_lstm,X_t,sc_fit):
    y_pred_test_lstm = model_lstm.predict(X_t)
    res = sc_fit.inverse_transform(y_pred_test_lstm).reshape(1, -1)
    return res

def lstm_analyze(dt,predict_len,predict_step):
    # TODO:输入输出格式一致
    is_date_index = 0
    if isinstance(dt,pd.DataFrame):
        is_date_index = 1
    elif isinstance(dt,pd.Series):
        is_date_index = 2
    last_date = dt.index[-1]
    if is_date_index:
        dt = dt.values
    timestamp = 100
    sc = MinMaxScaler(feature_range=(-1,1))
    sc_fit = sc.fit(dt[:,np.newaxis])
    lstm_model = lstm_train(dt,predict_len,sc_fit,timestamp)
    X_t = dt.values[-timestamp:]
    predict_res = lstm_predict(lstm_model,X_t,sc_fit).reshape(-1,1)
    if is_date_index:
        if predict_step.lower() == 'd':
            oldest_index = last_date + datetime.timedelta(days= 1)
        else:
            oldest_index = last_date + datetime.timedelta(hours= 1)
        index = pd.date_range(oldest_index,periods=predict_len,freq=predict_step.upper())
        if is_date_index == 1:
            predict_res = pd.DataFrame(data = np.array(predict_res),index=index)
        else:
            predict_res = pd.Series(data = np.array(predict_res),index=index)
    return predict_res
