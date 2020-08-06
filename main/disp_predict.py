#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/16

from utils.mysql_util import mysql_conn
from utils.hbase_util import *
import datetime
from stl_analyze import *
from lstm_analyze import *

def get_data_from_hbase(mac,predict_step):
    current_time = datetime.datetime.today()
    if predict_step == 'D':
        time_delta = datetime.timedelta(days=100)
        query_time = str(current_time - time_delta)[:19]
    else:
        time_delta = datetime.timedelta(hours=1000)
        query_time = str(current_time - time_delta)[:19]
    table = connection.table('mytable')

def get_data_from_http(mac,predict_step):
    current_time = datetime.datetime.today()
    if predict_step == 'D':
        time_delta = datetime.timedelta(days=100)
        query_time = str(current_time-time_delta)[:19]
    else:
        time_delta = datetime.timedelta(hours=1000)
        query_time = str(current_time-time_delta)[:19]
    return 1


def get_data_from_mysql(mac,predict_step):
    # TODO:
    current_time = datetime.datetime.today()
    if predict_step == 'D':
        time_delta = datetime.timedelta(days=100)
        query_time = str(current_time-time_delta)[:19]
    else:
        time_delta = datetime.timedelta(hours=1000)
        query_time = str(current_time-time_delta)[:19]
    get_data_from_mysql(mac,query_time)
    sql1 = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
           'bdmc.dataGnss where mac = "%s" and recordTime > "%s"' % (mac,query_time)
    data = pd.read_sql(sql1, mysql_conn).resample(predict_step).mean().interpolate()
    return data


def res_to_json(data):
    dt = pd.DataFrame(data)
    json_data = dt.to_json(orient='index',date_format='epoch',date_unit = 's')
    return json_data


def res_to_mysql(data):
    pass


def res_to_hbase(data):
    pass


def get_device_by_poi_id(poi_id):
    pass


def predict_gnss(mac,predict_step,predict_method,predict_len):
    data = get_data_from_mysql(mac, predict_step)
    predict_res = pd.DataFrame()
    for item in data.columns:
        if predict_method == 'arma':
            res = stl_arma_analyze(data[item], predict_len, predict_step)
        else:
            res = lstm_analyze(data[item], predict_len, predict_step)
        predict_res = pd.concat([predict_res,res],axis=1)
    predict_res.columns = data.columns
    return predict_res


def predict_crack(mac,predict_step,predict_method,predict_len):
    data = get_data_from_mysql(mac, predict_step)
    predict_res = pd.DataFrame()
    for item in data.columns:
        if predict_method == 'arma':
            res = stl_arma_analyze(data[item], predict_len, predict_step)
        else:
            res = lstm_analyze(data[item], predict_len, predict_step)
        predict_res = pd.concat([predict_res,res],axis=1)
    predict_res.columns = data.columns
    return predict_res


def predict_device(mac,predict_step,predict_method,predict_len):
    # 3D还是1D，这个需要考虑
    if mac == '0003%':
        data = predict_gnss(mac,predict_step,predict_method,predict_len)
    else:
        data = predict_crack(mac,predict_step,predict_method,predict_len)
    # TODO:
    ans = res_to_json(data)
    return ans



def predict_pois(poi_id,predict_step,predict_method,prdict_len):

    pass


def predict_project(project_id,predict_step,predict_method,predict_len):
    pass

if __name__ == '__main__':
    pass