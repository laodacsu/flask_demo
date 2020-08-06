import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
import warnings


url = 'http://cloud.bdsmc.net:8006/devicedata?mac=000300000079&num=1000'
data = pd.read_json(url)
warnings.filterwarnings("ignore")


prd_interval = 7
rolling_range_set = -100
rolling_range = -1 * int(abs(rolling_range_set) / prd_interval) * prd_interval
looking_range = 100
prd_window = -800
wavelet_type = 'db4'
wavelet_len = 4
level = 2
mode = 'sym'

prd_res = np.array(data['z'].values)[:rolling_range]
cur_prd = np.array([])


def wavelet_analyze(data):
    for interval in range(1, int(abs(rolling_range) / prd_interval) + 1):
        if rolling_range+prd_interval*interval == 0:
            index_for_predict = np.array(data['gps_time'])[rolling_range+prd_interval*(interval-1):]
            data_list2 = np.array(data['z'])[rolling_range+prd_interval*(interval-1):]
        else:
            index_for_predict = np.array(data['gps_time'])[rolling_range+prd_interval*(interval-1):rolling_range+prd_interval*interval]
            data_list2 = np.array(data['z'])[rolling_range+prd_interval*(interval-1):rolling_range+prd_interval*interval]
        index_list = np.array(data['gps_time'])[prd_window+interval*prd_interval:rolling_range + prd_interval * interval - prd_interval]
        data_list1 = np.array(data['z'])[prd_window+interval*prd_interval:rolling_range + prd_interval * interval - prd_interval]
        #     print(len(index_list))
        coeff = pywt.wavedec(data_list1, wavelet_type, mode=mode, level=level)

        # 选择模型阶数
        order = []
        model = []
        result = []
        Is_diff=np.zeros(len(coeff))
        for item_index,item in enumerate(coeff):
            dftest = adfuller(item)
            if dftest[1] > 0.05:
                print(dftest)
                print(len(item))
                temp_item = pd.DataFrame(item)
                base = temp_item.diff(1)
                base.dropna(inplace=True)
                Is_diff[item_index] = 1
                item = base.T.values[0]
                print(adfuller(item))
            else:
                base = item
            order_A2 = sm.tsa.arma_order_select_ic(base, ic='aic')['aic_min_order']
            order.append(order_A2)

            # 模型构建
            model_A2 = ARMA(base, order=order_A2)
            model.append(model_A2)

            results_A2 = model_A2.fit()
            result.append(results_A2)

        wave_len = len(index_list) + prd_interval
        t_len = int((wave_len - 1) / 2 + wavelet_len)
        # 预测小波系数
        AD = []
        # cA2,cD2,cD1
        for level_index in range(level + 1):
            if t_len == len(coeff[-1 - level_index]):
                pAD = coeff[-1 - level_index]
            else:
                pAD = np.hstack([coeff[-1 - level_index],model[-1-level_index].predict(params=result[-1 - level_index].params, start=len(coeff[-1 - level_index]),end=t_len - 1)])
            if level_index < level - 1:
                t_len = int((t_len - 1) / 2 + wavelet_len)
            AD.append(pAD)
        AD.reverse()

        coeff_new = AD
        predict_wave = pywt.waverec(coeff_new, wavelet_type)
        if Is_diff[item_index]:
            predict_wave = pd.DataFrame(predict_wave)
            diff_shift_prd = base.shift(1)
            predict_wave = predict_wave.add(diff_shift_prd).values[-prd_interval:]
        else:
            predict_wave=predict_wave[-prd_interval:]
        prd_res = np.hstack([prd_res, predict_wave])
        cur_prd = np.hstack([cur_prd,predict_wave])


        # 10个预测值
        print(interval)
        temp_data_wt = np.array([data_list2,predict_wave,predict_wave-data_list2,(predict_wave-data_list2)/data_list2*100])
        print(np.shape(temp_data_wt.T))
        predict_wt = pd.DataFrame(temp_data_wt.T,index = index_for_predict,columns=['real_value','pre_value_wt','err_wt','err_rate_wt/%'])
        print(predict_wt)
    plt.figure(figsize=(15, 5))
    plt.plot(data['z'].values[-1 * looking_range:], 'b')
    plt.plot(prd_res[-1 * looking_range:], 'r')
    plt.show()