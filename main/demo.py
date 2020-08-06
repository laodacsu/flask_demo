#!/usr/bin/rnv python
# -*- coding:utf-8 -*-
# authoor:zjd time:2020/3/16

from utils.mysql_util import *
import pandas as pd
a = pd.DataFrame()
sql = 'select recordTime as gps_time,offsetEast as x,offsetNorth as y,offsetHeight as z from ' \
           'bdmc.dataGnss where mac = "%s" and recordTime > "%s"' % ('000300000116','2020-03-17 10:00:00')

data = pd.read_sql(sql,con=mysql_conn,index_col='gps_time').resample('1H').last().interpolate()
data = pd.concat([a,data],axis=1)
p = data.to_json(orient='index',date_format='iso',date_unit = 's')
print(p)