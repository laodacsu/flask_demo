from utils.mongodb_util import *
from datetime import datetime,timedelta
import pandas as pd
import matplotlib.pyplot as plt
from utils.mysql_util import *
import numpy as np
import pymap3d as pm
import pywt


r = 6378137


def compute_x(dt,lon,lat):
    return [r*np.cos(dt[lat])*np.cos(dt[lon]),r*np.cos(dt[lat])*np.sin(dt[lon]),r*np.sin(dt[lat])]


def use_pymap3d(dt,lon,lat,alt):
    xyz = (pm.geodetic2ecef(dt[lat],dt[lon],dt[alt]))
    return list(xyz)


def disp_query_mongo(mac,query_time,collection_name):
    tmpresult = mydb[collection_name]
    query = {"rovertag": mac,'datetime':{'$gt':query_time},'quality':1}
    query_res = tmpresult.find(query)
    # 若使用一下代码，pandas 将取不到数据，加载一次有效
    # for item in query_res.limit(10):
    #     print(item['lon'])
        # print()
    df = pd.DataFrame(list(query_res))
    df = df.set_index('datetime',drop=True)
    ts_utc = df.index.tz_localize('UTC')
    time_index = list(ts_utc.tz_convert('Asia/Shanghai'))
    # df['datetime'] = df['datetime'] + pd.DateOffset(hours = 8)
    # hour_offset = pd.offsets.Hour(8)
    sql = 'select lng,lat,altitude from devices where mac = "%s"' % mac
    data = pd.read_sql(sql=sql, con=mysql_conn)
    # xyz = np.array(df.apply(compute_x,axis=1,args=('lon','lat')))
    # x0y0z0 = data.apply(compute_x,axis=1,args=('lng','lat'))
    xyz = np.array([[item[0],item[1],item[2]] for item in df.apply(use_pymap3d,axis=1,args=('lon','lat','alt'))])
    x0y0z0 = np.array(pm.geodetic2ecef(data['lat'].values,data['lng'].values,data['altitude'].values)).reshape(1,-1)
    xyz_res = (xyz-x0y0z0)*1000
    xyz_res = pd.DataFrame(data=xyz_res,columns=['x','y','z'],index=time_index)
    # 将数据添加至df中
    # df = pd.concat([df,xyz_res],axis=1)
    return xyz_res


def disp_query_mysql(mac,query_time,table_name):
    query_sql = 'select mac,recordTime,offsetEast,offsetNorth,offsetHeight from {} where' \
                ' mac = "{}" and recordTime > "{}"'.format(table_name,mac,
                                                           query_time.strftime("%Y-%m-%d %H:%M:%S"))
    data = pd.read_sql(query_sql, con=mysql_conn, index_col='recordTime')
    return data


def data_plot(data):
    data.plot(subplots=True,layout=(3, 1),sharex = True,style = '--*')


if __name__ == "__main__":
    mac = '00030000024e'
    query_time = datetime(2019,12,24,3,0,0)
    collection_name = 'tempResult'

    # mongo_data = disp_query_mongo(mac,query_time,collection_name)
    # data_plot(mongo_data)

    mysql_data = disp_query_mysql(mac,query_time,'dataGnss')
    data_plot(mysql_data)
    # plt.figure()
    # plt.plot(mongo_data.index,mongo_data['x'].values)
    # plt.scatter(mongo_data.index,mongo_data['x'].values)
    # plt.show()
    # predict_


