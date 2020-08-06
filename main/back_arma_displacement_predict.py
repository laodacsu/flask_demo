import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.interpolate import lagrange
from statsmodels.tsa.arima_model import ARMA,ARIMA
from datetime import datetime
import statsmodels.api as sm
from utils.mysql_util import *


def arma_predict(data):
    day_data = data.resample('1D').last()['x']
    # day_data.index = pd.to_datetime(day_data.index)
    # plt.plot(day_data.index.values,day_data.values)
    # plt.xticks(rotation=50)
    # plt.show()

    print(np.shape(day_data)[0])
    day_std = (day_data.std())
    day_mean = day_data.mean()

    # day_data.index.astype('0')
    dt = day_data.interpolate()
    # dt = day_data.fillna(method='ffill')
    # f = plt.figure(facecolor='white')
    # ax1 = f.add_subplot(211)
    # # plot_acf(dt, lags=31, ax=ax1)
    # ax2 = f.add_subplot(212)
    # plot_pacf(dt, lags=31, ax=ax2)
    # plt.show()
    # dftest = adfuller(dt)
    # print(dftest)


    pmax = int((dt.shape[0])/10) #一般阶数不超过length/10
    qmax = int((dt.shape[0])/10) #一般阶数不超过length/10

    # diff_num = 0
    # flag = True
    # while flag:
    #     t = adfuller(dt.values)
    #     if t[2] > 0.05:
    #         flag = False

    order_A2 = sm.tsa.arma_order_select_ic(dt,ic='aic')['aic_min_order']


    df = dt[:-5,]
    model = ARIMA(df,(1,1,1)).fit()
    ans = model.predict(start=len(df),end=len(dt)-1)
    # ans = model.forecast(5)
    ans = ans.cumsum()+dt.values[len(df)-1]

    plt.plot(dt.index,dt.values,'r')
    # plt.plot(dt.index,ans[0],'g'
    plt.plot(ans.index,ans.values)
    plt.legend(['True','predict'])
    plt.xticks(rotation=50)
    plt.show()

    dt_diff = dt.diff(1)
    dt_diff.dropna(inplace=True)


    order_A2 = sm.tsa.arma_order_select_ic(dt,ic='aic')['aic_min_order']
    model_A2 = ARMA(dt_diff,order=order_A2)
    results_A2 = model_A2.fit()
    print(results_A2.forcast(3))
    #
    #
    prd = model_A2.predict(params=results_A2.params,start=1,end=len(dt_diff+10))
    # prd.cumsum(axis=1)
    res = pd.DataFrame(index=dt_diff.index.values,data=prd)
    diff_shift_ts = dt.shift(1)
    diff_shift_ts.dropna(inplace=True)
    diff_recover_1 = res.values.reshape(res.values.shape[0],1)+diff_shift_ts.values.reshape(res.values.shape[0],1)

    plt.figure(facecolor='white')

    plt.plot(diff_shift_ts.index,diff_recover_1,'r')
    plt.plot(dt.index,dt.values,'g')
    plt.show()
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(diff_recover_1,dt.values[:-1]))


if __name__ == "__main__":
    sql = 'select recordTime as gps_time,offsetHeight as z,offsetEast as x,offsetNorth as y from' \
          ' bdmc.dataGnss where mac = "%s" and recordTime > "2019-6-16"' % ('000300000079')
    # url = 'http://cloud.bdsmc.net:8006/devicedata?mac=%s&num=1500'%'000300000079'
    # data = pd.read_json(url).set_index('gps_time')
    data = pd.read_sql(sql, mysql_conn, index_col='gps_time')
    arma_predict(data)