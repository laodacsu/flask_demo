import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.mysql_util import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.interpolate import lagrange
from statsmodels.tsa.arima_model import ARMA,ARIMA
from datetime import datetime
import statsmodels.api as sm


def gnss_day_data_acq(data):
    hour_data=data[['gps_time','x']].set_index('gps_time')
    daily_data = {}
    temp_date = hour_data.index[0]
    cur_date = hour_data.index[-1]
    day_range = (cur_date - temp_date).days
    for day_index in range(1,day_range):
        temp = hour_data[(temp_date+(day_index-1)*np.timedelta64(1,'D')):(temp_date+day_index*np.timedelta64(1,'D'))]
        if temp.shape[0]:
            daily_data[datetime.strptime(str((temp_date+(day_index-1)*np.timedelta64(1,'D')))[:10],'%Y-%m-%d')] = temp.mean()['x']
        else:
            daily_data[datetime.strptime(str((temp_date+(day_index-1)*np.timedelta64(1,'D')))[:10],'%Y-%m-%d')] = np.nan
    return pd.Series(daily_data)

def arma_predict(data):

    day_data = data.resample('1D').last()['x']
    # day_data.index = pd.to_datetime(day_data.index)
    plt.plot(day_data.index.values,day_data.values)
    plt.xticks(rotation=50)
    plt.show()
    print(np.shape(day_data)[0])
    day_std = (day_data.std())
    day_mean = day_data.mean()

    # day_data.index.astype('0')
    dt = day_data.interpolate()
    # # dt = day_data.fillna(method='ffill')
    # f = plt.figure(facecolor='white')
    # ax1 = f.add_subplot(211)
    # plot_acf(dt, lags=31, ax=ax1)
    # ax2 = f.add_subplot(212)
    # plot_pacf(dt, lags=31, ax=ax2)
    # plt.show()
    # dftest = adfuller(dt)
    # print(dftest)


    pmax = int((dt.shape[0])/10) #一般阶数不超过length/10
    qmax = int((dt.shape[0])/10) #一般阶数不超过length/10

    dt_diff = dt.diff(1)
    dt_diff.dropna(inplace=True)

    # #
    # bic_matrix = [] #bic矩阵
    # # print(day_data)
    # for p in range(pmax+1):
    #   tmp = []
    #   for q in range(qmax+1):
    #         try:
    #             tmp.append(ARIMA(dt_diff, (p,1,q)).fit().bic)
    #         except:
    #             tmp.append(np.NaN)
    #   bic_matrix.append(tmp)
    # bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值
    # print(bic_matrix)
    # p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
    # print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

    model = ARIMA(dt,(0,1,1)).fit()
    # model_A2
    ans = model.predict(start=1,end=len(dt))
    # model.
    plt.plot(ans.index,ans.values)
    plt.show()

    order_A2 = sm.tsa.arma_order_select_ic(dt_diff,ic='aic')['aic_min_order']
    model_A2 = ARMA(dt_diff,order=order_A2)
    results_A2 = model_A2.fit()
    #
    #
    prd = model_A2.predict(params=results_A2.params,start=1,end=len(dt_diff))
    res = pd.DataFrame(index=dt_diff.index.values,data=prd)
    diff_shift_ts = dt.shift(1)
    diff_shift_ts.dropna(inplace=True)
    diff_recover_1 = res.values.reshape(res.values.shape[0],1)+(diff_shift_ts).values.reshape(res.values.shape[0],1)
    #
    # print(type(res))

    plt.figure(facecolor='white')

    plt.plot(diff_shift_ts.index,diff_recover_1,'r')
    plt.plot(dt.index,dt.values,'g')
    plt.show()
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(diff_recover_1,dt.values[:-1]))

    # model = ARIMA(dt.values,order=(0,1,1)).fit()
    # prd = model.predict()
    # print(type(prd))
    # plt.figure(facecolor='white')
    # plt.plot(dt.index[:-1],prd,'r')
    # plt.plot(dt.index[:-1],dt.values[:-1],'g')
    # plt.show()

    # order_A2 = sm.tsa.arma_order_select_ic(dt_diff.values,ic='aic')['aic_min_order']
    # model_A2 = ARMA(dt_diff.values,order=order_A2)
    # results_A2 = model_A2.fit()
    # pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(dt_diff.values))
    # plt.plot(dt_diff.values)
    # plt.plot(pA2,'r')
    # plt.show()
    #
    # res = pd.DataFrame(index=dt_diff.index.values,data=pA2)
    # res_shift = dt.shift(1)
    # res_recover = res.add(res_shift)
    # plt.plot(dt.values)
    # plt.plot(res_recover,'r')
    # plt.show()


sql = 'select recordTime as gps_time,offsetHeight as z,offsetEast as x,offsetNorth as y from bdmc.dataGnss where mac = "%s" and recordTime > "2019-7-6"'%('000300000079')
# url = 'http://cloud.bdsmc.net:8006/devicedata?mac=%s&num=1500'%'000300000079'
# data = pd.read_json(url).set_index('gps_time')
data = pd.read_sql(sql, mysql_conn, index_col='gps_time')
plt.plot(data.index,data['x'].values)
plt.show()
arma_predict(data)