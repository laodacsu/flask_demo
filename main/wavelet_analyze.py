import pandas as pd
import numpy as np
import pywt
import arma_predict_displacement
from utils.mysql_util import *

wavelet_type = 'db4'
wavelet_len = 4
level = 2
mode = 'sym'


def wavelet_analyze(data,wavelet=None,mode=None,level = level):
    dt = []
    for dimension in range(data.shape[1]):
        cAD = pywt.wavedec(data[dimension].values,wavelet=wavelet,mode=mode,level=level)
        dt.append(cAD)
    return dt


def wavelet_predict(data,predict_step):
    if isinstance(data,pd.DataFrame) or isinstance(data,pd.Series):
        data = data.values
    avg = data[0]
    data = np.diff(data)
    dt = wavelet_analyze(data,wavelet=wavelet_type,mode=mode,level=level)
    predict_dt = []
    for item in dt:
        predict_len = data + predict_step
        for cAD,idx in enumerate(item[::-1]):
            cAD_predict_len = int((predict_len+1)/2) - np.array(cAD).shape[0]
            res,_,_,_ = arma_predict_displacement.arma_anlayze(cAD, cAD_predict_len)
            predict_len = int((predict_len+1)/2)
            predict_dt.append(res)
    predict_dt.reverse()
    predict_ans = pywt.waverec(predict_dt,wavelet=wavelet_type,mode=mode)
    return predict_ans


if __name__ == "__main__":
    sql = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
          'bdmc.dataGnss where mac = "%s" and recordTime > "2020-1-22 9:00:00"' % ('000300000116')
    data = pd.read_sql(sql, mysql_conn, index_col='gps_time').dropna()
    predict_ans = wavelet_predict(data.values,predict_step=21)

    print(1)
