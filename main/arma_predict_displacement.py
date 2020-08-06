import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from utils.mysql_util import *


def arma_train(dt,predict_len):
    '''
    arma 预测函数
    :param dt: 需预测位移数据
    :param predict_len: 预测长度
    :return: 预测模型结果
    '''
    # 差分次数
    diff_num = 0
    # 稳定标志位，True 不稳定  False 稳定
    flag = True
    base_values = []
    white_noise_flag = False
    can_arma_predict = True
    if isinstance(dt,pd.DataFrame) or isinstance(dt,pd.Series):
        data = dt.values
    else:
        data = dt
    # 判断序列是否稳定
    while flag:
        # 时间序列稳定性检验，单位根检验
        t_stable_confidence = adfuller(data, regression='c')[1]
        if t_stable_confidence < 0.1:
            flag = False
            # 若稳定则停止差分
            # 若不稳定，则继续差分
            continue
        base_values.append(data[-1])
        data = np.diff(data)
        diff_num += 1
    try:
        # 白噪声检验，检验序列是否属于白噪声，
        if acorr_ljungbox(data, lags=1)[1][0] > 0.5:
            # 若是白噪声，则取序列平均值做预测值
            white_noise_flag = True
            order_A2 = (0,0)
        else:
            # 若不是，则采用arma进行预测
            # 使用自动选择解算的函数进行阶数选择，使用aic准则评价
            # 训练数据长度为n-predict_len
            order_A2 = sm.tsa.arma_order_select_ic(data, ic=['aic'])['aic_min_order']
    except:
        # 在模型阶数选择或是训练时，可能存在模型不收敛的情况，此时，使用AR模型做预测
        order_A2 = (0, 1)
    if not white_noise_flag:
        avg = np.mean(data[-predict_len:])
        try:
            # 训练
            model = ARMA(data, order=order_A2).fit(disp=False)
        except:
            order_A2 = (0, order_A2[1])
            model = ARMA(data, order=order_A2).fit(disp=False)
            # 一般原order出错，则使用此方式同样出错
            can_arma_predict = False
    else:
        # 已经进行了差分
        avg = 0.0
        df = np.random.randn(200)
        model = ARMA(df, order=(1, 0)).fit(disp=False)
    return model,base_values,white_noise_flag,can_arma_predict,avg

def arma_predict_1(model,base_values,predict_len):
    '''
    arma 预测函数
    :param model: 训练出的arma预测模型
    :param base_values: 差分初值
    :param predict_len: 预测长度
    :return:
    '''
    diff_num  = base_values.__len__()
    forcast_res = model.forecast(predict_len)
    # model.predict()
    frocast_value = forcast_res[0]
    down_border = forcast_res[2][:, 0]
    upper_border = forcast_res[2][:, 1]
    ans = frocast_value
    if diff_num:
        for diff in range(diff_num):
            diff_base_value = base_values[diff_num - diff - 1]
            ans = ans.cumsum() + diff_base_value
            frocast_value = frocast_value.cumsum() + diff_base_value
            down_border = down_border.cumsum() + diff_base_value
            upper_border = upper_border.cumsum() + diff_base_value
    return ans,frocast_value,down_border,upper_border

def arma_predict_2(predict_len,avg,base_values):
    '''
    arma无法使用情况下处理
    :param predict_len:
    :param avg:
    :param base_values:
    :return:
    '''
    diff_num = base_values.__len__()
    ans = avg * np.ones([predict_len, 1])
    for diff in range(diff_num):
        diff_base_value = base_values[diff_num - diff - 1]
        ans = ans.cumsum() + diff_base_value
    frocast_value = ans
    down_border = ans - 1
    upper_border = ans + 1
    return ans,frocast_value,down_border,upper_border


def arma_i(dt,predict_len):
    # TODO：判断数据类型，对应输出
    model, base_values, white_noise_flag,can_arma_predict,avg = arma_train(dt,predict_len)
    if can_arma_predict and not white_noise_flag:
        ans, frocast_value, down_border, upper_border = arma_predict_1(model, base_values,predict_len)
    else:
        ans, frocast_value, down_border, upper_border = arma_predict_2(predict_len,avg,base_values)
    return ans,frocast_value,down_border,upper_border,model


def arma_anlayze(dt, predict_len):
    '''
    :param dt: ndArray，
    :param predict_len:  预测长度
    :return: array包括 预测结果 预测结果 预测结果下界 预测结果上界
    '''
    if isinstance(dt,pd.DataFrame) or isinstance(dt,pd.Series):
        temp_dt = dt.values
    else:
        temp_dt = dt
    # 差分次数
    diff_num = 0
    # 稳定标志位，True 不稳定  False 稳定
    flag = True
    base_values = []
    white_noise_flag = False
    # 判断序列是否稳定
    while flag:
        # 时间序列稳定性检验，单位根检验
        t_stable_confidence = adfuller(temp_dt, regression='c')[1]
        if t_stable_confidence < 0.1:
            flag = False
            # 若稳定则停止差分
            # 若不稳定，则继续差分
            continue
        base_values.append(temp_dt[-1])
        temp_dt = np.diff(temp_dt)
        diff_num += 1
    try:
        # 白噪声检验，检验序列是否属于白噪声，
        if acorr_ljungbox(temp_dt, lags=1)[1][0] > 0.5:
            # 若是白噪声，则取序列平均值做预测值
            white_noise_flag = True
            order_A2 = (0,0)
        else:
            # 若不是，则采用arma进行预测
            # 使用自动选择解算的函数进行阶数选择，使用aic准则评价
            # 训练数据长度为n-predict_len
            order_A2 = sm.tsa.arma_order_select_ic(temp_dt, ic=['aic'])['aic_min_order']
    except:
        # 在模型阶数选择或是训练时，可能存在模型不收敛的情况，此时，使用AR模型做预测
        order_A2 = (0, 1)
        # print('the order is :' + str(order_A2))
    if not white_noise_flag:
        try:
            # 训练
            model = ARMA(temp_dt, order=order_A2).fit(disp=False)
        except:
            # 一般原order出错，则使用此方式同样出错
            order_A2 = (0,order_A2[1])
            model = ARMA(temp_dt, order=order_A2).fit(disp=False)
            flag = True
        avg = np.mean(temp_dt[-predict_len:])
        # 预测
        if not flag:
            # forecasts are produced using the predict method from a results instance.
            forcast_res = model.forecast(predict_len)
            # model.predict()
            frocast_value = forcast_res[0]
            down_border = forcast_res[2][:,0]
            upper_border = forcast_res[2][:,1]
            ans = frocast_value
            for diff in range(diff_num):
                diff_base_value = base_values[diff_num - diff - 1]
                ans = ans.cumsum() + diff_base_value
                frocast_value = frocast_value.cumsum() + diff_base_value
                down_border = down_border.cumsum() + diff_base_value
                upper_border = upper_border.cumsum() + diff_base_value
        #    为不稳定序列时处理方法
        else:
            ans = avg * np.ones([predict_len, 1])
            for diff in range(diff_num):
                diff_base_value = base_values[diff_num - diff - 1]
                ans = ans.cumsum() + diff_base_value
            frocast_value = ans
            down_border = ans - 1
            upper_border = ans + 1
    #  为白噪声序列时处理方法
    else:
        # 已经进行了差分
        avg = 0.0
        ans = avg * np.ones([predict_len,1])
        df = np.random.randn(200)
        model = ARMA(df, order=(1, 0)).fit(disp=False)
        for diff in range(diff_num):
            diff_base_value = base_values[diff_num - diff - 1]
            ans = ans.cumsum() + diff_base_value
        frocast_value = ans
        down_border = ans-1
        upper_border = ans+1
    return [ans.reshape(-1,1),frocast_value,down_border,upper_border,model]

def evaluation(real_data,predict_data):
    '''
    :param real_data: 实际数据
    :param predict_data: 预测数据
    :return:
    '''
    # 模型评估
    if isinstance(real_data,pd.Series):
        real_data = real_data.values
    if isinstance(predict_data,pd.DataFrame):
        predict_data = predict_data.values.flatten()
    rmse = np.sqrt(mean_squared_error(real_data, predict_data))  # 均方误差
    mae = mean_absolute_error(real_data, predict_data)  # 绝对误差
    F_norm = np.linalg.norm(real_data - predict_data) / np.linalg.norm(real_data)
    r2 = 1 - ((real_data - predict_data) ** 2).sum() / ((real_data - real_data.mean()) ** 2).sum()
    var = 1 - (np.var(real_data - predict_data)) / np.var(real_data)
    return rmse, mae, 1 - F_norm, r2, var


def gnss_predict(data, predict_len = 21):
    '''
    gnss位移数据预测
    :param data: 数据，DataFrame
    :param predict_len: 预测长度
    :return: 无返回
    '''
    # 数据长度不够则不预测
    if data.shape[0] > predict_len:
        # data.interpolate()
        # data.plot()
        # plt.show()
        plt.style.use('ggplot')
        plt.figure()
        for index, dimension in enumerate(['x', 'y', 'z']):
            # 以天为单位，取平均值，缺失天数采用插值法
            dt = data.resample('1H').mean()[dimension].interpolate()
            plt.subplot(3, 1, index + 1)
            # 预测
            predict_dt = dt[:-predict_len]
            ans = arma_anlayze(predict_dt, predict_len)
            date_index = pd.date_range(predict_dt.index[-1], periods=predict_len+1, freq='H')

            ans[0] = pd.DataFrame(data=ans[0],index=date_index[1:])
            ans[1] = pd.DataFrame(data=ans[1],index=date_index[1:])
            ans[2] = pd.DataFrame(data=ans[2],index=date_index[1:])
            ans[3] = pd.DataFrame(data=ans[3],index=date_index[1:])

            if ans is None:
                continue
            plt.title(dimension)
            plt.plot(dt.index[:], dt.values[:], 'r-*')
            # plt.plot(dt.index,ans[0],'g')
            plt.plot(ans[1].index, ans[1].values, 'g--o')
            plt.legend(['True', 'predict'])
            plt.xticks(rotation=50)
            # plt.figure()
            # plt.plot(ans[1].index,ans[1])
            plt.fill_between(ans[1].index,ans[2].values[:,0],ans[3].values[:,0],alpha=0.2)
            rmse, mae, accuracy, r2, var = evaluation(dt[-predict_len:], ans[1])
            print('{:-^30}'.format(index+1))
            print("RMSE: {},\nMAE: {},\nAccuracy: {},\nr2:{},\nVar: {}".format(rmse, mae, accuracy, r2, var))
        plt.show()


if __name__ == "__main__":
    sql = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
          'bdmc.dataGnss where mac = "%s" and recordTime > "2020-1-22 9:00:00"' % ('000300000116')
    data = pd.read_sql(sql, mysql_conn, index_col='gps_time').dropna()
    # data.plot()
    gnss_predict(data, predict_len = 21)
    # plt.figure()
    diff_data = data.resample('1H').mean().interpolate().diff(1).dropna()
    data.resample('1H').mean().interpolate().diff(1).dropna().plot(subplots=True,layout=(3,1),sharex = True,style = '--*')
    plt.show()