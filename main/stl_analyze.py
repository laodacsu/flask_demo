import pandas as pd
from utils.mysql_util import *
import arma_predict_displacement
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import datetime
from PyEMD import EMD
emd = EMD()

import warnings
warnings.filterwarnings("ignore")


def plot_data_no_index(raw_dt, dt, tmp):
    if isinstance(tmp,pd.DataFrame) or isinstance(tmp,pd.Series):
        tmp = tmp.values.flatten().tolist()
    plt.figure()
    dt = dt.tolist()
    dt.extend(tmp)
    dt = np.array(dt)
    plt.plot(raw_dt)
    plt.plot(dt, 'g--o')
    plt.legend(('real', 'predict'))
    plt.xlabel('time')
    plt.ylabel('displacement')
    plt.show()

def plot_data_index(raw_dt,dt,tmp):
    dx1 = pd.DataFrame(raw_dt)
    # todo：
    dx2 = pd.DataFrame(pd.concat([dt,tmp]))
    dx = pd.concat([dx1,dx2],axis=1)
    dx.columns=['real','predict']
    dx.plot()
    plt.show()


# train
# return 模型训练结果
def stl_arma_analyze(data, predict_len,predict_step):
    # TODO：STL强制要求输入为带时间的
    is_date_index = 1
    if not isinstance(data,pd.DataFrame) and not isinstance(data,pd.Series):
        is_date_index = 0
        lastest_date_index = datetime.datetime.today()
        oldest_index = lastest_date_index - pd.Timedelta(predict_len, unit=predict_step.upper())
        index = pd.date_range(oldest_index, periods=predict_len, freq=predict_step.upper())
        data = pd.DataFrame(data = data,index=index)
    stl = STL(data)
    res = stl.fit()
    it = 0
    tmp = np.zeros((predict_len,1))
    res = [res.seasonal,res.trend]
    for index,item in enumerate(res):
        it = it + 1
        ans, _, _, _,_ = arma_predict_displacement.arma_i(item.values, predict_len=predict_len)
        tmp = tmp + ans.reshape(-1,1)
    # tmp = np.array(tmp).sum(axis=1)
    res = tmp.flatten()
    if is_date_index:
        if predict_step.lower() == 'd':
            oldest_index = data.index[-1] + datetime.timedelta(days= 1)
        else:
            oldest_index = data.index[-1] + datetime.timedelta(hours= 1)
        index = pd.date_range(oldest_index,periods=predict_len,freq=predict_step.upper())
        if is_date_index == 1:
            res = pd.DataFrame(data = np.array(res),index=index)
        else:
            res = pd.Series(data = np.array(res),index=index)
    return res


def stl_holt_analyze(data, predict_len,predict_step):
    # TODO：STL强制要求输入为带时间的
    is_date_index = 1
    if not isinstance(data,pd.DataFrame) and not isinstance(data,pd.Series):
        is_date_index = 0
        lastest_date_index = datetime.datetime.today()
        oldest_index = lastest_date_index - pd.Timedelta(predict_len, unit=predict_step.upper())
        index = pd.date_range(oldest_index, periods=predict_len, freq=predict_step.upper())
        data = pd.DataFrame(data = data,index=index)
    stl = STL(data)
    res = stl.fit()
    it = 0
    tmp = np.zeros((predict_len,1))
    res = [res.seasonal,res.trend]
    for index,item in enumerate(res):
        it = it + 1
        # fit1 = SimpleExpSmoothing(item.values).fit()
        fit1 = Holt(item.values).fit(smoothing_level= 0.8)
        # fit1 = ExponentialSmoothing(item.values).fit(smoothing_level= 0.2)
        ans = fit1.forecast(predict_len)
        # ans, _, _, _,_ = arma_predict_displacement.arma_i(item.values, predict_len=predict_len)
        tmp = tmp + ans.reshape(-1,1)
    # tmp = np.array(tmp).sum(axis=1)
    res = tmp.flatten()
    if is_date_index:
        if predict_step.lower() == 'd':
            oldest_index = data.index[-1] + datetime.timedelta(days= 1)
        else:
            oldest_index = data.index[-1] + datetime.timedelta(hours= 1)
        index = pd.date_range(oldest_index,periods=predict_len,freq=predict_step.upper())
        if is_date_index == 1:
            res = pd.DataFrame(data = np.array(res),index=index)
        else:
            res = pd.Series(data = np.array(res),index=index)
    return res


def holt_analyze(data,predict_len):
    fit1 = Holt(data).fit(smoothing_level=0.8)
    # fit1 = ExponentialSmoothing(item.values).fit(smoothing_level= 0.2)
    ans = fit1.forecast(predict_len)
    return ans


if __name__ == "__main__":
    time1 = datetime.datetime.now()
    mac = '000300000079'
    sql1 = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
          'bdmc.dataGnss where mac = "%s" and recordTime > "2020-2-28 8:00:00"' % (mac)
    data_1 = pd.read_sql(sql1, mysql_conn, index_col='gps_time').dropna().resample('1H').mean().interpolate()
    # predict_disp = stl_arma_analyze(data_1['x'][:-24], 24,'H')
    # predict_disp = holt_analyze(data_1['x'][:-24],24)
    predict_disp = stl_holt_analyze(data_1['x'][:-24], 24,'H')
    # plot_data_index(data_1['x'], data_1['x'][:-24], predict_disp)
    # plot_data_no_index(data_1['x'].values, data_1['x'].values[:-21], predict_disp)

    time2 = datetime.datetime.now()
    sql2 = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
          'bdmc.dataGnss where mac = "%s" and recordTime > "2019-9-22 8:00:00"' % (mac)
    data_2 = pd.read_sql(sql2, mysql_conn, index_col='gps_time').dropna().resample('1D').mean().interpolate()
    predict_disp2 = holt_analyze(data_2['x'][:-7],7)
    # predict_disp2 = stl_holt_analyze(data_2['x'][:-7], 7,'D')
    # plot_data_index(data_2['x'], data_2['x'][:-7], predict_disp2)

    # predict_disp2 = stl_arma_analyze(data_2['x'][:-7], 7,'D')
    # plot_data_no_index(data_2['x'].values, data_2['x'].values[:-21], predict_disp2)
    # plot_data_index(data_2['x'], data_2['x'][:-7], predict_disp2)
    time3 = datetime.datetime.now()
    print('小时预测耗时 %s 秒'%(time2-time1).seconds)
    print('天预测耗时 %s 秒'%(time3-time2).seconds)
